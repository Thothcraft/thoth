"""
Federated Learning components for CSI-based activity recognition.

Contains:
- FedXgbBagging: Federated XGBoost bagging aggregation strategy
- Helper functions for FL simulation with FederatedPartitioner
- Verbose logging utilities
"""

import json
import time
from collections.abc import Callable
from logging import WARNING
from typing import Any, cast

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =============================================================================
# FedXgbBagging Strategy (from Flower)
# =============================================================================
def aggregate_xgb_trees(bst_prev_org: bytes | None, bst_curr_org: bytes) -> bytes:
    """Conduct bagging aggregation for given XGBoost trees.
    
    Combines trees from multiple clients by appending them to the global model.
    
    Parameters
    ----------
    bst_prev_org : bytes or None
        Previous global model as bytes. None for first round.
    bst_curr_org : bytes
        Current client model as bytes.
    
    Returns
    -------
    bytes
        Aggregated model.
    """
    if not bst_prev_org:
        return bst_curr_org

    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)

    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")
    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> tuple[int, int]:
    """Get number of trees and parallel trees from XGBoost model."""
    xgb_model = json.loads(bytearray(xgb_model_org))
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


# =============================================================================
# FL Simulation for XGBoost with FederatedPartitioner
# =============================================================================
class FedXGBoostSimulator:
    """Simulates Federated XGBoost training using FederatedPartitioner.
    
    Implements FedXgbBagging strategy in a simulation environment.
    Each partition represents a client with local data.
    
    Parameters
    ----------
    partitioner : FederatedPartitioner
        Partitioner containing the federated data splits.
    num_rounds : int
        Number of federated rounds. Default: 5
    local_epochs : int
        Number of local boosting rounds per client per FL round. Default: 1
    xgb_params : dict
        XGBoost parameters. Default: sensible defaults for classification.
    test_dataset : TrainingDataset, optional
        Held-out test set for global evaluation.
    verbose : bool
        Enable verbose logging. Default: True
    """
    
    def __init__(
        self,
        partitioner,
        num_rounds=5,
        local_epochs=1,
        xgb_params=None,
        test_dataset=None,
        verbose=True,
    ):
        self.partitioner = partitioner
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.test_dataset = test_dataset
        self.verbose = verbose
        
        self.xgb_params = xgb_params or {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'max_depth': 4,
            'eta': 0.1,
            'num_parallel_tree': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        
        self.global_model: bytes | None = None
        self.history = {
            'round': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'test_f1': [],
            'num_trees': [],
            'round_time': [],
        }
    
    def _log(self, msg, level='INFO'):
        """Print verbose log message."""
        if self.verbose:
            print(f"[FL-XGB] [{level}] {msg}")
    
    def _log_separator(self, char='=', length=70):
        if self.verbose:
            print(char * length)
    
    def _client_train(self, partition_id: int, global_round: int) -> tuple[bytes, int]:
        """Train a single client and return updated model.
        
        Parameters
        ----------
        partition_id : int
            Client/partition ID.
        global_round : int
            Current FL round (1-indexed).
        
        Returns
        -------
        tuple[bytes, int]
            (local_model_bytes, num_samples)
        """
        partition = self.partitioner.load_partition(partition_id)
        X, y = partition.X, partition.y
        num_samples = len(X)
        
        # Ensure num_class is set for multi-class (use global class count, not local)
        params = self.xgb_params.copy()
        num_classes = len(self.partitioner.dataset.label_map)
        if num_classes > 2:
            params['num_class'] = num_classes
        
        dtrain = xgb.DMatrix(X, label=y)
        
        if global_round == 1:
            # First round: train from scratch
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=self.local_epochs,
                verbose_eval=False,
            )
        else:
            # Subsequent rounds: load global model and continue training
            bst = xgb.Booster(params=params)
            bst.load_model(bytearray(self.global_model))
            
            # Local training: update with new trees
            for _ in range(self.local_epochs):
                bst.update(dtrain, bst.num_boosted_rounds())
            
            # Extract only the new trees for aggregation
            bst = bst[
                bst.num_boosted_rounds() - self.local_epochs : bst.num_boosted_rounds()
            ]
        
        local_model = bst.save_raw("json")
        return local_model, num_samples
    
    def _aggregate_round(self, client_models: list[tuple[bytes, int]]) -> None:
        """Aggregate client models using bagging.
        
        Parameters
        ----------
        client_models : list of (model_bytes, num_samples)
        """
        for model_bytes, _ in client_models:
            self.global_model = aggregate_xgb_trees(self.global_model, model_bytes)
    
    def _evaluate_global(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate global model on given data."""
        if self.global_model is None:
            return {'accuracy': 0.0, 'f1': 0.0}
        
        # Load global model
        params = self.xgb_params.copy()
        num_classes = len(np.unique(y))
        if num_classes > 2:
            params['num_class'] = num_classes
            
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(self.global_model))
        
        dtest = xgb.DMatrix(X)
        y_pred = bst.predict(dtest)
        
        # Handle multi-class vs binary
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = y_pred.astype(int)
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        return {'accuracy': acc, 'f1': f1, 'predictions': y_pred}
    
    def run(self) -> dict:
        """Run the federated learning simulation.
        
        Returns
        -------
        dict
            Training history and final metrics.
        """
        num_clients = self.partitioner.num_partitions
        
        self._log_separator()
        self._log(f"Starting Federated XGBoost Simulation")
        self._log_separator()
        self._log(f"Clients: {num_clients}")
        self._log(f"FL Rounds: {self.num_rounds}")
        self._log(f"Local Epochs: {self.local_epochs}")
        self._log(f"XGB Params: {self.xgb_params}")
        self._log_separator('-')
        
        total_start = time.time()
        
        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()
            self._log(f"\n{'='*20} ROUND {round_num}/{self.num_rounds} {'='*20}")
            
            # Client training
            client_models = []
            total_samples = 0
            
            for client_id in range(num_clients):
                client_start = time.time()
                model_bytes, num_samples = self._client_train(client_id, round_num)
                client_time = time.time() - client_start
                
                client_models.append((model_bytes, num_samples))
                total_samples += num_samples
                
                self._log(
                    f"  Client {client_id}: {num_samples} samples, "
                    f"trained in {client_time:.2f}s",
                    level='DEBUG' if not self.verbose else 'INFO'
                )
            
            # Aggregation
            agg_start = time.time()
            self._aggregate_round(client_models)
            agg_time = time.time() - agg_start
            
            # Get tree count
            if self.global_model:
                num_trees, _ = _get_tree_nums(self.global_model)
            else:
                num_trees = 0
            
            self._log(f"  Aggregated {num_clients} clients in {agg_time:.2f}s")
            self._log(f"  Global model: {num_trees} trees")
            
            # Evaluation
            train_acc = 0.0
            if round_num == self.num_rounds or round_num % max(1, self.num_rounds // 3) == 0:
                # Evaluate on all training data
                all_X = []
                all_y = []
                for cid in range(num_clients):
                    part = self.partitioner.load_partition(cid)
                    all_X.append(part.X)
                    all_y.append(part.y)
                X_train_all = np.concatenate(all_X)
                y_train_all = np.concatenate(all_y)
                
                train_metrics = self._evaluate_global(X_train_all, y_train_all)
                train_acc = train_metrics['accuracy']
                self._log(f"  Train Accuracy: {train_acc:.4f}")
            
            test_acc = 0.0
            test_f1 = 0.0
            if self.test_dataset is not None:
                test_metrics = self._evaluate_global(self.test_dataset.X, self.test_dataset.y)
                test_acc = test_metrics['accuracy']
                test_f1 = test_metrics['f1']
                self._log(f"  Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            round_time = time.time() - round_start
            self._log(f"  Round time: {round_time:.2f}s")
            
            # Record history
            self.history['round'].append(round_num)
            self.history['train_accuracy'].append(train_acc)
            self.history['test_accuracy'].append(test_acc)
            self.history['test_f1'].append(test_f1)
            self.history['num_trees'].append(num_trees)
            self.history['round_time'].append(round_time)
        
        total_time = time.time() - total_start
        
        # Final evaluation
        self._log_separator()
        self._log("FINAL RESULTS")
        self._log_separator()
        
        final_metrics = {
            'total_time': total_time,
            'num_rounds': self.num_rounds,
            'num_clients': num_clients,
            'final_num_trees': self.history['num_trees'][-1] if self.history['num_trees'] else 0,
            'history': self.history,
        }
        
        if self.test_dataset is not None:
            test_metrics = self._evaluate_global(self.test_dataset.X, self.test_dataset.y)
            final_metrics['test_accuracy'] = test_metrics['accuracy']
            final_metrics['test_f1'] = test_metrics['f1']
            
            cm = confusion_matrix(self.test_dataset.y, test_metrics['predictions'])
            final_metrics['confusion_matrix'] = cm.tolist()
            
            self._log(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
            self._log(f"Final Test F1: {test_metrics['f1']:.4f}")
            self._log(f"Total Trees: {final_metrics['final_num_trees']}")
            self._log(f"Total Time: {total_time:.2f}s")
            self._log(f"Confusion Matrix:")
            for row in cm:
                self._log(f"  {row.tolist()}")
        
        self._log_separator()
        
        return final_metrics
    
    def get_global_model(self) -> xgb.Booster | None:
        """Get the final global XGBoost model."""
        if self.global_model is None:
            return None
        
        params = self.xgb_params.copy()
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(self.global_model))
        return bst


def run_federated_xgboost_experiment(
    dataset,
    test_dataset=None,
    num_partitions=5,
    alpha=0.5,
    num_rounds=5,
    local_epochs=1,
    xgb_params=None,
    verbose=True,
):
    """Run a complete federated XGBoost experiment.
    
    Convenience function that creates partitioner and runs simulation.
    
    Parameters
    ----------
    dataset : TrainingDataset
        Full training dataset to partition.
    test_dataset : TrainingDataset, optional
        Held-out test set.
    num_partitions : int
        Number of FL clients. Default: 5
    alpha : float
        Dirichlet concentration for non-IID partitioning. Default: 0.5
    num_rounds : int
        Number of FL rounds. Default: 5
    local_epochs : int
        Local training epochs per round. Default: 1
    xgb_params : dict, optional
        XGBoost parameters.
    verbose : bool
        Enable verbose output. Default: True
    
    Returns
    -------
    dict
        Experiment results including metrics and history.
    """
    from utils import FederatedPartitioner
    
    if verbose:
        print(f"\n{'='*70}")
        print("EXPERIMENT C: Federated XGBoost with Dirichlet Partitioning")
        print(f"{'='*70}")
    
    # Create partitioner
    partitioner = FederatedPartitioner(
        dataset=dataset,
        num_partitions=num_partitions,
        alpha=alpha,
        seed=42,
    )
    
    if verbose:
        print(f"Created {num_partitions} partitions with alpha={alpha}")
        for i in range(num_partitions):
            part = partitioner.load_partition(i)
            unique, counts = np.unique(part.y, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"  Partition {i}: {len(part.X)} samples, distribution: {dist}")
    
    # Run simulation
    simulator = FedXGBoostSimulator(
        partitioner=partitioner,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        xgb_params=xgb_params,
        test_dataset=test_dataset,
        verbose=verbose,
    )
    
    results = simulator.run()
    results['partitioner'] = partitioner
    results['simulator'] = simulator
    
    return results


# =============================================================================
# FedAvg Simulator for MLP
# =============================================================================
class FedAvgMLPSimulator:
    """Simulates Federated Averaging with MLP models.

    Each client trains a local MLP; the server averages model weights.

    Parameters
    ----------
    partitioner : FederatedPartitioner
    num_rounds : int
    local_epochs : int
    hidden_dims : list of int
    dropout : float
    lr : float
    batch_size : int
    test_dataset : TrainingDataset, optional
    verbose : bool
    """

    def __init__(self, partitioner, num_rounds=5, local_epochs=2,
                 hidden_dims=None, dropout=0.3, lr=1e-3, batch_size=64,
                 test_dataset=None, verbose=True):
        import torch
        self.partitioner = partitioner
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.test_dataset = test_dataset
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_state = None
        self.history = {'round': [], 'test_accuracy': [], 'test_f1': [], 'round_time': []}

    def _log(self, msg):
        if self.verbose:
            print(f"[FL-Avg] {msg}")

    def _make_model(self, n_features, n_classes):
        from dl import MLP
        return MLP(n_features, self.hidden_dims, n_classes, dropout=self.dropout)

    def run(self):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from dl import MLP
        import copy

        num_clients = self.partitioner.num_partitions
        part0 = self.partitioner.load_partition(0)
        n_features = part0.X.shape[1]
        n_classes = len(self.partitioner.dataset.label_map)

        global_model = self._make_model(n_features, n_classes).to(self.device)
        self.global_state = copy.deepcopy(global_model.state_dict())

        self._log("=" * 70)
        self._log(f"Starting FedAvg MLP Simulation")
        self._log(f"Clients: {num_clients}, Rounds: {self.num_rounds}, Local epochs: {self.local_epochs}")
        self._log(f"Architecture: {self.hidden_dims}, LR: {self.lr}")
        self._log("-" * 70)

        total_start = time.time()

        for rnd in range(1, self.num_rounds + 1):
            rnd_start = time.time()
            self._log(f"\n{'='*20} ROUND {rnd}/{self.num_rounds} {'='*20}")

            client_states = []
            client_sizes = []

            for cid in range(num_clients):
                part = self.partitioner.load_partition(cid)
                X_c, y_c = part.X, part.y
                client_sizes.append(len(y_c))

                local_model = self._make_model(n_features, n_classes).to(self.device)
                local_model.load_state_dict(copy.deepcopy(self.global_state))

                ds = TensorDataset(torch.FloatTensor(X_c), torch.LongTensor(y_c))
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
                optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
                criterion = nn.CrossEntropyLoss()

                local_model.train()
                for _ in range(self.local_epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        optimizer.zero_grad()
                        loss = criterion(local_model(xb), yb)
                        loss.backward()
                        optimizer.step()

                client_states.append(copy.deepcopy(local_model.state_dict()))
                self._log(f"  Client {cid}: {len(y_c)} samples")

            # Weighted average
            total_samples = sum(client_sizes)
            avg_state = {}
            for key in self.global_state:
                avg_state[key] = sum(
                    client_states[i][key].float() * (client_sizes[i] / total_samples)
                    for i in range(num_clients)
                )
            self.global_state = avg_state

            # Evaluate
            test_acc, test_f1_val = 0.0, 0.0
            if self.test_dataset is not None:
                eval_model = self._make_model(n_features, n_classes).to(self.device)
                eval_model.load_state_dict(self.global_state)
                eval_model.eval()
                with torch.no_grad():
                    xte = torch.FloatTensor(self.test_dataset.X).to(self.device)
                    preds = eval_model(xte).argmax(dim=1).cpu().numpy()
                test_acc = accuracy_score(self.test_dataset.y, preds)
                test_f1_val = f1_score(self.test_dataset.y, preds, average='weighted', zero_division=0)
                self._log(f"  Test Accuracy: {test_acc:.4f}, F1: {test_f1_val:.4f}")

            rnd_time = time.time() - rnd_start
            self._log(f"  Round time: {rnd_time:.2f}s")
            self.history['round'].append(rnd)
            self.history['test_accuracy'].append(test_acc)
            self.history['test_f1'].append(test_f1_val)
            self.history['round_time'].append(rnd_time)

        total_time = time.time() - total_start

        # Final eval
        final = {'total_time': total_time, 'num_rounds': self.num_rounds,
                 'num_clients': num_clients, 'history': self.history}
        if self.test_dataset is not None:
            eval_model = self._make_model(n_features, n_classes).to(self.device)
            eval_model.load_state_dict(self.global_state)
            eval_model.eval()
            with torch.no_grad():
                xte = torch.FloatTensor(self.test_dataset.X).to(self.device)
                preds = eval_model(xte).argmax(dim=1).cpu().numpy()
            final['test_accuracy'] = accuracy_score(self.test_dataset.y, preds)
            final['test_f1'] = f1_score(self.test_dataset.y, preds, average='weighted', zero_division=0)
            cm = confusion_matrix(self.test_dataset.y, preds)
            final['confusion_matrix'] = cm.tolist()

            self._log("=" * 70)
            self._log("FINAL RESULTS")
            self._log(f"  Test Accuracy: {final['test_accuracy']:.4f}")
            self._log(f"  Test F1:       {final['test_f1']:.4f}")
            self._log(f"  Total Time:    {total_time:.2f}s")
            self._log(f"  Confusion Matrix:")
            for row in cm:
                self._log(f"    {row.tolist()}")
            self._log("=" * 70)

        return final


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import load_csi_datasets, FederatedPartitioner, TrainingDataset

    TRAIN_DIR = '../../../wifi_sensing_data/thoth_data/train'
    TEST_DIR  = '../../../wifi_sensing_data/thoth_data/test'
    WINDOW_LEN = 1500
    NUM_PARTITIONS = 5
    ALPHA = 1e6  # IID partitioning (very high alpha -> uniform distribution)
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 5

    print("=" * 70)
    print("FL EXPERIMENTS: FedXGBoost vs FedAvg")
    print("=" * 70)

    combined_ds, test_ds = load_csi_datasets(TRAIN_DIR, TEST_DIR, WINDOW_LEN, verbose=False)
    print(f"Train: {combined_ds.X.shape}, Test: {test_ds.X.shape}")
    print(f"Classes: {combined_ds.num_classes}, Label map: {combined_ds.label_map}")

    xgb_params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'eta': 0.1,
        'num_parallel_tree': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # Time-series split helper
    def time_split(ds, test_frac=0.2):
        X, y = ds.X, ds.y
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            sp = len(ci) - max(1, int(len(ci) * test_frac))
            train_idx.append(ci[:sp])
            test_idx.append(ci[sp:])
        tri = np.concatenate(train_idx)
        tei = np.concatenate(test_idx)
        # Build sub-datasets
        train_sub = TrainingDataset.__new__(TrainingDataset)
        train_sub.dataset_files = []
        train_sub.feature_key = ds.feature_key
        train_sub.label_map = ds.label_map
        train_sub.balance = False
        train_sub._X = X[tri]
        train_sub._y = y[tri]
        test_sub = TrainingDataset.__new__(TrainingDataset)
        test_sub.dataset_files = []
        test_sub.feature_key = ds.feature_key
        test_sub.label_map = ds.label_map
        test_sub.balance = False
        test_sub._X = X[tei]
        test_sub._y = y[tei]
        return train_sub, test_sub

    all_results = {}

    for exp_name, use_separate in [ ("B: Separate test set", True), ("A: Time-series split", False)]:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_name}")
        print(f"{'='*70}")

        if use_separate:
            train_data, eval_data = combined_ds, test_ds
        else:
            train_data, eval_data = time_split(combined_ds)

        print(f"  Train: {train_data.X.shape[0]}, Test: {eval_data.X.shape[0]}")

        partitioner = FederatedPartitioner(
            dataset=train_data, num_partitions=NUM_PARTITIONS, alpha=ALPHA, seed=42)

        print(f"  Partitions ({NUM_PARTITIONS}, alpha={ALPHA}):")
        for i in range(NUM_PARTITIONS):
            p = partitioner.load_partition(i)
            unique, counts = np.unique(p.y, return_counts=True)
            print(f"    Client {i}: {len(p.X)} samples, dist={dict(zip(unique.tolist(), counts.tolist()))}")

        exp_res = {}

        # --- FedXGBoost ---
        print(f"\n  --- FedXGBoost (Bagging) ---")
        sim_xgb = FedXGBoostSimulator(
            partitioner=partitioner, num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
            xgb_params=xgb_params, test_dataset=eval_data, verbose=True)
        exp_res['FedXGB'] = sim_xgb.run()

        # --- FedAvg MLP ---
        print(f"\n  --- FedAvg (MLP) ---")
        sim_avg = FedAvgMLPSimulator(
            partitioner=partitioner, num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
            hidden_dims=[256, 128], dropout=0.3, lr=1e-3, batch_size=64,
            test_dataset=eval_data, verbose=True)
        exp_res['FedAvg'] = sim_avg.run()

        all_results[exp_name] = exp_res

    # ---- Final comparison ----
    print(f"\n{'='*80}")
    print("FINAL COMPARISON: FedXGBoost vs FedAvg")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} | {'Strategy':<10} | {'Acc':>7} {'F1':>7} | {'Time':>7}")
    print("-" * 75)
    for exp_name, exp_res in all_results.items():
        for strat, m in exp_res.items():
            acc = m.get('test_accuracy', 0)
            f1v = m.get('test_f1', 0)
            t = m.get('total_time', 0)
            print(f"{exp_name:<30} | {strat:<10} | {acc:>7.4f} {f1v:>7.4f} | {t:>6.1f}s")

    print(f"\n{'='*70}")
    print("FL experiments completed!")
    print(f"{'='*70}")
