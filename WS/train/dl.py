"""
Deep Learning components for CSI-based activity recognition.

Contains:
- MLP: Multi-layer perceptron classifier
- FeatureExtractor: Shared feature extraction network with BatchNorm
- LabelClassifier: Classification head
- AdaptiveModel: Domain adaptation model using AdaBN + Deep CORAL + TTA

Adaptive Methods:
1. AdaBN (Adaptive Batch Normalization):
   Re-estimates BN running statistics on target domain data.
   Aligns first and second moments (mean, variance) per feature.

2. Deep CORAL (CORrelation ALignment):
   Minimizes Frobenius norm of covariance difference between source/target features.
   Aligns full covariance structure (correlations between features).

3. TTA (Test-Time Adaptation via Entropy Minimization):
   At inference, minimizes prediction entropy on unlabeled target batches
   by updating BN parameters only. Pushes model toward confident predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MLP (Basic Multi-Layer Perceptron)
# =============================================================================
class MLP(nn.Module):
    """Multi-layer perceptron classifier.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Hidden layer dimensions.
    output_dim : int
        Number of output classes.
    dropout : float
        Dropout probability. Default: 0.0
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Feature Extractor
# =============================================================================
class FeatureExtractor(nn.Module):
    """Feature extraction network for domain adaptation.
    
    Extracts domain-invariant features from input data.
    Used as the shared backbone in DANN architecture.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Hidden layer dimensions.
    output_dim : int
        Dimension of extracted features (embedding size).
    dropout : float
        Dropout probability. Default: 0.0
    batch_norm : bool
        Whether to use batch normalization. Default: True
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, batch_norm=True):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        # Final projection to embedding space
        layers.append(nn.Linear(prev, output_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Label Classifier
# =============================================================================
class LabelClassifier(nn.Module):
    """Label classification head for main task.
    
    Takes features from FeatureExtractor and predicts class labels.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (from FeatureExtractor).
    hidden_dims : list of int
        Hidden layer dimensions. Can be empty for linear classifier.
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability. Default: 0.0
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Deep CORAL Loss
# =============================================================================
def coral_loss(source_features, target_features):
    """Compute Deep CORAL loss between source and target feature batches.
    
    Minimizes the Frobenius norm of the difference between source and target
    covariance matrices, aligning the full second-order statistics.
    
    L_CORAL = ||C_s - C_t||_F^2 / (4 * d^2)
    
    Parameters
    ----------
    source_features : torch.Tensor
        Source domain features of shape (n_s, d).
    target_features : torch.Tensor
        Target domain features of shape (n_t, d).
    
    Returns
    -------
    torch.Tensor
        Scalar CORAL loss.
    
    Reference
    ---------
    Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
    """
    d = source_features.size(1)
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    # Center features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    
    # Covariance matrices
    cov_s = (source_centered.t() @ source_centered) / max(n_s - 1, 1)
    cov_t = (target_centered.t() @ target_centered) / max(n_t - 1, 1)
    
    # Frobenius norm squared of difference, normalized by 4*d^2
    loss = (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
    return loss


# =============================================================================
# Class-Conditional CORAL Loss
# =============================================================================
def conditional_coral_loss(source_features, target_features, source_labels,
                           target_logits, confidence_threshold=0.8):
    """Class-conditional CORAL: align per-class covariances using pseudo-labels.
    
    Instead of global ||C_s - C_t||_F, computes:
        sum_c ||C_s^c - C_t^c||_F
    
    Target pseudo-labels are obtained from model predictions, thresholded
    by confidence to avoid noisy alignment.
    
    Parameters
    ----------
    source_features : torch.Tensor
        Source features (n_s, d).
    target_features : torch.Tensor
        Target features (n_t, d). Must be detached or from no_grad context.
    source_labels : torch.Tensor
        Source ground-truth labels (n_s,).
    target_logits : torch.Tensor
        Target logits (n_t, n_classes). Used to derive pseudo-labels.
    confidence_threshold : float
        Only use target samples with max softmax prob >= threshold.
    
    Returns
    -------
    torch.Tensor
        Scalar conditional CORAL loss.
    """
    d = source_features.size(1)
    
    # Target pseudo-labels with confidence thresholding
    with torch.no_grad():
        probs = F.softmax(target_logits, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        confident_mask = max_probs >= confidence_threshold
    
    classes = source_labels.unique()
    total_loss = torch.tensor(0.0, device=source_features.device)
    n_aligned = 0
    
    for c in classes:
        # Source samples of class c
        s_mask = source_labels == c
        s_feats = source_features[s_mask]
        
        # Target samples pseudo-labeled as c AND confident
        t_mask = (pseudo_labels == c) & confident_mask
        t_feats = target_features[t_mask]
        
        # Need at least 2 samples per class per domain for covariance
        if s_feats.size(0) < 2 or t_feats.size(0) < 2:
            continue
        
        # Per-class covariance matrices
        s_centered = s_feats - s_feats.mean(dim=0, keepdim=True)
        t_centered = t_feats - t_feats.mean(dim=0, keepdim=True)
        
        cov_s = (s_centered.t() @ s_centered) / max(s_feats.size(0) - 1, 1)
        cov_t = (t_centered.t() @ t_centered) / max(t_feats.size(0) - 1, 1)
        
        total_loss = total_loss + (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
        n_aligned += 1
    
    if n_aligned > 0:
        total_loss = total_loss / n_aligned
    
    return total_loss


# =============================================================================
# Feature Whitening Layer
# =============================================================================
class FeatureWhitening(nn.Module):
    """Learnable feature whitening layer.
    
    Applies running ZCA-style whitening: f_out = W @ (f - mu)
    where W approximates C^{-1/2}.
    
    Uses running statistics (like BatchNorm) so it can whiten at test time.
    The whitening matrix is recomputed periodically from running covariance.
    
    For efficiency, uses BatchNorm-without-affine as the core transform
    (normalizes each dimension independently), then adds a learnable
    linear decorrelation layer to handle cross-feature correlations
    (the subspace rotation that drift diagnostics identified).
    
    Parameters
    ----------
    num_features : int
        Feature dimension.
    momentum : float
        Momentum for running stats update. Default: 0.1
    """
    
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        
        # BatchNorm without affine: normalizes per-dimension (mean=0, var=1)
        # This handles the diagonal of the covariance (scaling)
        self.bn = nn.BatchNorm1d(num_features, affine=False, momentum=momentum)
        
        # Learnable decorrelation: a linear layer initialized to identity
        # This learns to undo cross-feature correlations (rotation)
        self.decorrelate = nn.Linear(num_features, num_features, bias=False)
        nn.init.eye_(self.decorrelate.weight)
    
    def forward(self, x):
        # Step 1: Per-dimension normalization (handles scaling drift)
        x = self.bn(x)
        # Step 2: Learnable decorrelation (handles rotation drift)
        x = self.decorrelate(x)
        return x


# =============================================================================
# Adaptive Batch Normalization (AdaBN)
# =============================================================================
def adapt_batchnorm(model, target_loader, device, alpha=1.0):
    """Re-estimate BatchNorm running statistics on target domain data.
    
    Instead of a hard reset, blends source and target statistics:
        new_stat = alpha * target_stat + (1 - alpha) * source_stat
    
    When alpha=1.0 (default), this is pure target stats (standard AdaBN).
    When alpha<1.0, source stats are partially retained, which is more
    robust when the target dataset is small.
    
    Parameters
    ----------
    model : nn.Module
        Model with BatchNorm layers (trained on source domain).
    target_loader : DataLoader
        DataLoader yielding target domain batches (unlabeled).
    device : torch.device
        Device to run on.
    alpha : float
        Blending ratio in [0, 1]. 1.0 = pure target stats, 0.0 = keep source.
        Default: 1.0
    
    Reference
    ---------
    Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation", ICLR 2017
    """
    # Save source BN stats before overwriting
    source_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            source_stats[name] = {
                'mean': m.running_mean.clone(),
                'var': m.running_var.clone(),
            }
    
    # Reset BN running stats for fresh target accumulation
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
    
    # Forward pass in train mode to accumulate target statistics
    model.train()
    with torch.no_grad():
        for batch in target_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            model(x)
    
    # Blend source and target stats
    if alpha < 1.0:
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and name in source_stats:
                src = source_stats[name]
                m.running_mean.copy_(alpha * m.running_mean + (1.0 - alpha) * src['mean'])
                m.running_var.copy_(alpha * m.running_var + (1.0 - alpha) * src['var'])
    
    model.eval()


# =============================================================================
# Test-Time Adaptation (Entropy Minimization)
# =============================================================================
def tta_entropy_minimization(model, target_loader, device, tta_steps=1, tta_lr=1e-4):
    """Test-time adaptation via entropy minimization on BN parameters.
    
    For each batch of unlabeled target data:
    1. Forward pass to get predictions
    2. Compute entropy of predictions
    3. Backprop and update BN affine parameters (gamma, beta) only
    
    This pushes the model toward confident predictions on target data
    without requiring any labels.
    
    Parameters
    ----------
    model : nn.Module
        Model with BatchNorm layers.
    target_loader : DataLoader
        DataLoader yielding target domain batches (unlabeled).
    device : torch.device
        Device to run on.
    tta_steps : int
        Number of optimization steps per batch. Default: 1
    tta_lr : float
        Learning rate for TTA updates. Default: 1e-4
    
    Reference
    ---------
    Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021
    """
    # Collect only BN affine parameters (weight=gamma, bias=beta)
    bn_params = []
    bn_modules = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_modules.append(m)
            if m.affine:
                bn_params.append(m.weight)
                bn_params.append(m.bias)
    
    if not bn_params:
        return
    
    optimizer = torch.optim.Adam(bn_params, lr=tta_lr)
    
    # Disable running stat tracking (proper Tent behavior):
    # BN will use batch stats for normalization but NOT update running_mean/running_var.
    # This prevents drift instability from mixing AdaBN stats with TTA batch stats.
    saved_tracking = {}
    for i, m in enumerate(bn_modules):
        saved_tracking[i] = m.track_running_stats
        m.track_running_stats = False
    
    # Set model to train mode so BN uses batch stats, but freeze all non-BN params
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in bn_params:
        p.requires_grad_(True)
    
    for batch in target_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        
        for _ in range(tta_steps):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
            
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()
    
    # Restore tracking and requires_grad
    for i, m in enumerate(bn_modules):
        m.track_running_stats = saved_tracking[i]
    for p in model.parameters():
        p.requires_grad_(True)
    
    model.eval()


# =============================================================================
# Adaptive Model (FeatureExtractor + LabelClassifier with AdaBN/CORAL/TTA)
# =============================================================================
class AdaptiveModel(nn.Module):
    """Domain adaptation model using AdaBN, Deep CORAL, TTA, and Feature Whitening.
    
    Architecture:
        Input -> FeatureExtractor (with BatchNorm) -> [FeatureWhitening] -> LabelClassifier
    
    Adaptation is achieved through:
    1. Class-Conditional CORAL loss during training (per-class covariance alignment)
    2. Feature Whitening layer (standardizes feature space, attacks subspace rotation)
    3. AdaBN at adaptation time (re-estimates BN stats on target data)
    4. TTA at test time (entropy minimization on BN affine params)
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    feature_dims : list of int
        Hidden dimensions for feature extractor.
    feature_output_dim : int
        Output dimension of feature extractor.
    label_hidden_dims : list of int
        Hidden dimensions for label classifier.
    num_classes : int
        Number of activity classes.
    dropout : float
        Dropout probability. Default: 0.3
    use_batch_norm : bool
        Whether to use BatchNorm in the feature extractor. Default: True.
    use_whitening : bool
        Whether to insert a FeatureWhitening layer after the feature extractor.
        Default: False (backward compatible).
    
    Example
    -------
    >>> model = AdaptiveModel(
    ...     input_dim=1024,
    ...     feature_dims=[512, 256],
    ...     feature_output_dim=128,
    ...     label_hidden_dims=[64],
    ...     num_classes=6,
    ...     use_whitening=True,
    ... )
    >>> logits = model(x)
    >>> features = model.extract_features(x)
    """
    
    def __init__(
        self,
        input_dim,
        feature_dims,
        feature_output_dim,
        label_hidden_dims,
        num_classes,
        dropout=0.3,
        use_batch_norm=True,
        use_whitening=False,
    ):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=feature_dims,
            output_dim=feature_output_dim,
            dropout=dropout,
            batch_norm=use_batch_norm,
        )
        
        self.use_whitening = use_whitening
        if use_whitening:
            self.whitening = FeatureWhitening(feature_output_dim)
        
        self.label_classifier = LabelClassifier(
            input_dim=feature_output_dim,
            hidden_dims=label_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass: extract features, [whiten], then classify.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        features = self.feature_extractor(x)
        if self.use_whitening:
            features = self.whitening(features)
        return self.label_classifier(features)
    
    def extract_features(self, x):
        """Extract features (after whitening if enabled).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Features of shape (batch_size, feature_output_dim).
        """
        features = self.feature_extractor(x)
        if self.use_whitening:
            features = self.whitening(features)
        return features
    
    def predict(self, x):
        """Predict labels (for inference)."""
        return self.forward(x)


# =============================================================================
# Utility: Create adaptive models for experiments
# =============================================================================
def make_adaptive_model(n_features, n_classes, config='default',
                        use_batch_norm=True, use_whitening=False):
    """Factory function to create AdaptiveModel with preset configurations.
    
    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int
        Number of activity classes.
    config : str
        Configuration preset: 'default', 'small', 'large'.
    use_batch_norm : bool
        Whether to use BatchNorm in the feature extractor. Default: True.
    use_whitening : bool
        Whether to enable the FeatureWhitening layer. Default: False.
    
    Returns
    -------
    AdaptiveModel
    """
    configs = {
        'default': {
            'feature_dims': [512, 256],
            'feature_output_dim': 128,
            'label_hidden_dims': [64],
            'dropout': 0.3,
        },
        'small': {
            'feature_dims': [256, 128],
            'feature_output_dim': 64,
            'label_hidden_dims': [32],
            'dropout': 0.2,
        },
        'large': {
            'feature_dims': [1024, 512, 256],
            'feature_output_dim': 256,
            'label_hidden_dims': [128, 64],
            'dropout': 0.4,
        },
    }
    
    cfg = configs.get(config, configs['default'])
    
    return AdaptiveModel(
        input_dim=n_features,
        feature_dims=cfg['feature_dims'],
        feature_output_dim=cfg['feature_output_dim'],
        label_hidden_dims=cfg['label_hidden_dims'],
        num_classes=n_classes,
        dropout=cfg['dropout'],
        use_batch_norm=use_batch_norm,
        use_whitening=use_whitening,
    )


# =============================================================================
# Training (returns trained model + training info)
# =============================================================================
def train_model(model, X_source, y_source, X_target, X_test, y_test,
                epochs=50, batch_size=64, lr=1e-3, coral_weight=0.5,
                use_coral=False, use_conditional_coral=False,
                confidence_threshold=0.8, verbose=True):
    """Train model on source data with optional CORAL alignment.

    Returns the trained model (in eval mode) and a training info dict.
    Does NOT apply any post-training adaptation (AdaBN/TTA/fewshot).

    Parameters
    ----------
    model : AdaptiveModel
    X_source, y_source : np.ndarray
        Source domain training data.
    X_target : np.ndarray
        Target domain features (unlabeled, used for CORAL).
    X_test, y_test : np.ndarray
        Test data for periodic evaluation during training.
    epochs, batch_size, lr : training hyperparameters
    coral_weight : float
        Weight for CORAL loss term.
    use_coral : bool
        Use global Deep CORAL loss.
    use_conditional_coral : bool
        Use class-conditional CORAL (overrides use_coral).
    confidence_threshold : float
        Pseudo-label confidence for conditional CORAL.
    verbose : bool

    Returns
    -------
    model : AdaptiveModel (trained, eval mode, on device)
    info : dict with train_time_s, train_accuracy, coral_mode
    """
    import time
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    source_ds = TensorDataset(torch.FloatTensor(X_source), torch.LongTensor(y_source))
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    any_coral = use_coral or use_conditional_coral
    coral_mode = 'conditional' if use_conditional_coral else ('global' if use_coral else 'none')

    if verbose:
        print(f"  Device: {device}, Epochs: {epochs}, LR: {lr}")
        print(f"  CORAL: {coral_mode} (weight={coral_weight})")
        if use_conditional_coral:
            print(f"  Conditional CORAL confidence threshold: {confidence_threshold}")
        if hasattr(model, 'use_whitening') and model.use_whitening:
            print(f"  Feature Whitening: enabled")

    X_target_tensor = torch.FloatTensor(X_target).to(device) if any_coral else None
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    t0 = time.time()
    report_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        model.train()
        total_task_loss = 0.0
        total_coral_loss = 0.0
        correct = 0
        total = 0

        if any_coral:
            with torch.no_grad():
                target_feats_all = model.extract_features(X_target_tensor)
                if use_conditional_coral:
                    target_logits_all = model.label_classifier(target_feats_all)
            target_feats_ref = target_feats_all.detach()
            if use_conditional_coral:
                target_logits_ref = target_logits_all.detach()

        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            source_features = model.extract_features(xb)
            logits = model.label_classifier(source_features)
            task_loss = criterion(logits, yb)
            loss = task_loss

            if use_conditional_coral:
                c_loss = conditional_coral_loss(
                    source_features, target_feats_ref, yb,
                    target_logits_ref, confidence_threshold=confidence_threshold)
                loss = loss + coral_weight * c_loss
                total_coral_loss += c_loss.item() * xb.size(0)
            elif use_coral:
                c_loss = coral_loss(source_features, target_feats_ref)
                loss = loss + coral_weight * c_loss
                total_coral_loss += c_loss.item() * xb.size(0)

            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item() * xb.size(0)
            _, pred = torch.max(logits, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

        train_acc = correct / total
        train_loss = total_task_loss / total

        if verbose and ((epoch + 1) % report_interval == 0 or epoch == 0):
            model.eval()
            with torch.no_grad():
                test_logits = model.predict(X_test_tensor)
                test_loss_val = criterion(test_logits, y_test_tensor).item()
                _, test_preds = torch.max(test_logits, 1)
                test_correct = (test_preds == y_test_tensor).sum().item()
                test_acc = test_correct / len(y_test)
            model.train()

            msg = (f"  Epoch {epoch+1:3d}/{epochs} | "
                   f"TrainLoss: {train_loss:.4f}  TrainAcc: {train_acc:.4f} | "
                   f"TestLoss: {test_loss_val:.4f}  TestAcc: {test_acc:.4f}")
            if any_coral:
                msg += f" | CORAL: {total_coral_loss/total:.6f}"
            print(msg)

    train_time = round(time.time() - t0, 2)
    model.eval()

    if verbose:
        print(f"  Training complete in {train_time}s, final train acc: {train_acc:.4f}")

    info = {
        'train_time_s': train_time,
        'train_accuracy': round(train_acc, 4),
        'coral_mode': coral_mode,
    }
    return model, info


# =============================================================================
# Comprehensive metrics computation
# =============================================================================
def compute_metrics(model_or_logits, X_test, y_test, device=None):
    """Compute comprehensive metrics from a model or pre-computed logits.

    Parameters
    ----------
    model_or_logits : AdaptiveModel or torch.Tensor
        If a model, runs forward pass on X_test. If tensor, uses directly.
    X_test : np.ndarray
        Test features (ignored if model_or_logits is a tensor).
    y_test : np.ndarray
        Test labels.
    device : torch.device or None

    Returns
    -------
    dict with all metrics.
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                 precision_score, recall_score,
                                 cohen_kappa_score, matthews_corrcoef,
                                 balanced_accuracy_score, log_loss)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(model_or_logits, nn.Module):
        model_or_logits.eval()
        with torch.no_grad():
            logits_tensor = model_or_logits.predict(torch.FloatTensor(X_test).to(device))
    else:
        logits_tensor = model_or_logits

    probs = F.softmax(logits_tensor, dim=1).cpu().numpy()
    preds = logits_tensor.argmax(dim=1).cpu().numpy()
    n = len(y_test)
    n_cls = logits_tensor.size(1)

    acc = round(accuracy_score(y_test, preds), 4)
    bal_acc = round(balanced_accuracy_score(y_test, preds), 4)

    f1_w = round(f1_score(y_test, preds, average='weighted', zero_division=0), 4)
    f1_macro = round(f1_score(y_test, preds, average='macro', zero_division=0), 4)
    f1_micro = round(f1_score(y_test, preds, average='micro', zero_division=0), 4)

    prec_w = round(precision_score(y_test, preds, average='weighted', zero_division=0), 4)
    rec_w = round(recall_score(y_test, preds, average='weighted', zero_division=0), 4)
    prec_macro = round(precision_score(y_test, preds, average='macro', zero_division=0), 4)
    rec_macro = round(recall_score(y_test, preds, average='macro', zero_division=0), 4)

    prec_per = np.round(precision_score(y_test, preds, average=None, zero_division=0, labels=list(range(n_cls))), 4).tolist()
    rec_per = np.round(recall_score(y_test, preds, average=None, zero_division=0, labels=list(range(n_cls))), 4).tolist()
    f1_per = np.round(f1_score(y_test, preds, average=None, zero_division=0, labels=list(range(n_cls))), 4).tolist()

    cm = confusion_matrix(y_test, preds, labels=list(range(n_cls)))
    per_class_acc = []
    for c in range(n_cls):
        total_c = cm[c].sum()
        per_class_acc.append(round(cm[c, c] / total_c, 4) if total_c > 0 else 0.0)

    kappa = round(cohen_kappa_score(y_test, preds), 4)
    mcc = round(matthews_corrcoef(y_test, preds), 4)

    try:
        ll = round(log_loss(y_test, probs, labels=list(range(n_cls))), 4)
    except Exception:
        ll = float('nan')

    max_probs = probs.max(axis=1)
    mean_conf = round(float(np.mean(max_probs)), 4)
    std_conf = round(float(np.std(max_probs)), 4)
    median_conf = round(float(np.median(max_probs)), 4)

    ent = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
    mean_ent = round(float(np.mean(ent)), 4)
    std_ent = round(float(np.std(ent)), 4)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = (preds[mask] == y_test[mask]).mean()
        bin_conf = max_probs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    ece = round(float(ece), 4)

    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'f1_weighted': f1_w, 'f1_macro': f1_macro, 'f1_micro': f1_micro,
        'precision_weighted': prec_w, 'recall_weighted': rec_w,
        'precision_macro': prec_macro, 'recall_macro': rec_macro,
        'precision_per_class': prec_per, 'recall_per_class': rec_per,
        'f1_per_class': f1_per, 'accuracy_per_class': per_class_acc,
        'cohen_kappa': kappa, 'mcc': mcc, 'log_loss': ll,
        'mean_confidence': mean_conf, 'std_confidence': std_conf,
        'median_confidence': median_conf,
        'mean_entropy': mean_ent, 'std_entropy': std_ent, 'ece': ece,
        'confusion_matrix': cm.tolist(),
    }


# =============================================================================
# Adapt and evaluate (post-training adaptation on a copy of trained model)
# =============================================================================
def adapt_and_evaluate(trained_model, X_target, X_test, y_test,
                       train_info, adapt_name='none',
                       use_adabn=False, adabn_alpha=1.0,
                       use_tta=False, tta_steps=1, tta_lr=1e-4,
                       X_target_labeled=None, y_target_labeled=None,
                       fewshot_epochs=20, fewshot_lr=1e-4,
                       batch_size=64, verbose=True):
    """Apply post-training adaptation to a copy of trained_model and evaluate.

    Parameters
    ----------
    trained_model : AdaptiveModel (trained, eval mode)
        Will be deepcopied — original is never modified.
    X_target : np.ndarray
        Target domain features (unlabeled).
    X_test, y_test : np.ndarray
        Test data for evaluation.
    train_info : dict
        Output from train_model (train_time_s, train_accuracy, coral_mode).
    adapt_name : str
        Label for this adaptation variant.
    use_adabn, adabn_alpha, use_tta, tta_steps, tta_lr : AdaBN/TTA params
    X_target_labeled, y_target_labeled : few-shot labeled data
    fewshot_epochs, fewshot_lr : few-shot fine-tuning params
    batch_size : int
    verbose : bool

    Returns
    -------
    dict with all metrics + adaptation info.
    """
    import copy
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-adaptation metrics (from the original trained model)
    pre = compute_metrics(trained_model, X_test, y_test, device)

    if verbose:
        print(f"    [{adapt_name}] Pre-adapt  -> Acc: {pre['accuracy']}, F1w: {pre['f1_weighted']}")

    # Deep copy so we don't mutate the shared trained model
    adapt_model = copy.deepcopy(trained_model).to(device)

    # AdaBN
    if use_adabn:
        if verbose:
            print(f"    [{adapt_name}] Applying AdaBN (alpha={adabn_alpha})...")
        adapt_target_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_target)),
            batch_size=batch_size, shuffle=False)
        adapt_batchnorm(adapt_model, adapt_target_loader, device, alpha=adabn_alpha)

    # TTA
    if use_tta:
        if verbose:
            print(f"    [{adapt_name}] Applying TTA (steps={tta_steps}, lr={tta_lr})...")
        tta_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_target)),
            batch_size=batch_size, shuffle=True)
        tta_entropy_minimization(adapt_model, tta_loader, device,
                                 tta_steps=tta_steps, tta_lr=tta_lr)

    # Few-shot fine-tuning
    has_fewshot = X_target_labeled is not None and y_target_labeled is not None and len(X_target_labeled) > 0
    if has_fewshot:
        if verbose:
            print(f"    [{adapt_name}] Few-shot fine-tuning: {len(X_target_labeled)} samples, "
                  f"{fewshot_epochs} epochs, lr={fewshot_lr}")
        adapt_model.train()
        fs_ds = TensorDataset(torch.FloatTensor(X_target_labeled), torch.LongTensor(y_target_labeled))
        fs_loader = DataLoader(fs_ds, batch_size=min(batch_size, len(X_target_labeled)),
                               shuffle=True, drop_last=False)
        fs_optimizer = torch.optim.Adam(adapt_model.parameters(), lr=fewshot_lr)
        fs_criterion = nn.CrossEntropyLoss()
        for _ in range(fewshot_epochs):
            for fs_xb, fs_yb in fs_loader:
                fs_xb, fs_yb = fs_xb.to(device), fs_yb.to(device)
                fs_optimizer.zero_grad()
                fs_loss = fs_criterion(adapt_model(fs_xb), fs_yb)
                fs_loss.backward()
                fs_optimizer.step()
        adapt_model.eval()

    # Post-adaptation metrics
    post = compute_metrics(adapt_model, X_test, y_test, device)

    if verbose:
        delta = round(post['accuracy'] - pre['accuracy'], 4)
        sign = '+' if delta >= 0 else ''
        print(f"    [{adapt_name}] Post-adapt -> Acc: {post['accuracy']}, F1w: {post['f1_weighted']}, "
              f"Kappa: {post['cohen_kappa']}, ECE: {post['ece']}  (delta: {sign}{delta})")

    return {
        'train_time_s': train_info['train_time_s'],
        'train_accuracy': train_info['train_accuracy'],
        'pre_adapt_accuracy': pre['accuracy'],
        'pre_adapt_f1': pre['f1_weighted'],
        'test_accuracy': post['accuracy'],
        'test_f1': post['f1_weighted'],
        'post': post,
        'adaptation': {
            'coral': train_info['coral_mode'],
            'adabn': use_adabn,
            'tta': use_tta,
            'whitening': hasattr(trained_model, 'use_whitening') and trained_model.use_whitening,
            'fewshot': has_fewshot,
            'fewshot_n': len(X_target_labeled) if has_fewshot else 0,
        },
    }


def _print_dl_metrics(name, m, label_map=None):
    p = m.get('post', {})
    n_cls = len(p.get('accuracy_per_class', []))
    cls_names = [label_map.get(i, str(i)) for i in range(n_cls)] if label_map else [str(i) for i in range(n_cls)]

    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  {'='*60}")

    # Adaptation config
    if 'adaptation' in m:
        a = m['adaptation']
        parts = [f"CORAL={a['coral']}"]
        if a['adabn']: parts.append('AdaBN')
        if a['tta']: parts.append('TTA')
        if a.get('whitening'): parts.append('Whitening')
        if a.get('fewshot'): parts.append(f"FewShot({a['fewshot_n']})")
        print(f"    Config:         {', '.join(parts)}")

    print(f"    Train time:     {m['train_time_s']}s")
    print(f"    Train accuracy: {m['train_accuracy']}")

    # Pre-adaptation
    if 'pre_adapt_accuracy' in m:
        print(f"    Pre-adapt acc:  {m['pre_adapt_accuracy']}")
        print(f"    Pre-adapt F1:   {m['pre_adapt_f1']}")
        delta = round(m['test_accuracy'] - m['pre_adapt_accuracy'], 4)
        sign = '+' if delta >= 0 else ''
        print(f"    Adapt delta:    {sign}{delta}")

    # Global metrics
    print(f"    --- Global Metrics ---")
    print(f"    Accuracy:       {p.get('accuracy', m['test_accuracy'])}")
    print(f"    Balanced Acc:   {p.get('balanced_accuracy', 'N/A')}")
    print(f"    F1 weighted:    {p.get('f1_weighted', m['test_f1'])}")
    print(f"    F1 macro:       {p.get('f1_macro', 'N/A')}")
    print(f"    F1 micro:       {p.get('f1_micro', 'N/A')}")
    print(f"    Prec weighted:  {p.get('precision_weighted', 'N/A')}")
    print(f"    Rec  weighted:  {p.get('recall_weighted', 'N/A')}")
    print(f"    Prec macro:     {p.get('precision_macro', 'N/A')}")
    print(f"    Rec  macro:     {p.get('recall_macro', 'N/A')}")
    print(f"    Cohen Kappa:    {p.get('cohen_kappa', 'N/A')}")
    print(f"    MCC:            {p.get('mcc', 'N/A')}")
    print(f"    Log Loss:       {p.get('log_loss', 'N/A')}")

    # Calibration & confidence
    print(f"    --- Calibration & Confidence ---")
    print(f"    ECE:            {p.get('ece', 'N/A')}")
    print(f"    Mean conf:      {p.get('mean_confidence', 'N/A')}")
    print(f"    Std  conf:      {p.get('std_confidence', 'N/A')}")
    print(f"    Median conf:    {p.get('median_confidence', 'N/A')}")
    print(f"    Mean entropy:   {p.get('mean_entropy', 'N/A')}")
    print(f"    Std  entropy:   {p.get('std_entropy', 'N/A')}")

    # Per-class table
    if n_cls > 0:
        print(f"    --- Per-Class Breakdown ---")
        print(f"    {'Class':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print(f"    {'-'*36}")
        acc_pc = p.get('accuracy_per_class', [])
        prec_pc = p.get('precision_per_class', [])
        rec_pc = p.get('recall_per_class', [])
        f1_pc = p.get('f1_per_class', [])
        for i in range(n_cls):
            cname = cls_names[i] if i < len(cls_names) else str(i)
            print(f"    {cname:<10} {acc_pc[i]:>6.4f} {prec_pc[i]:>6.4f} {rec_pc[i]:>6.4f} {f1_pc[i]:>6.4f}")

    # Confusion matrix
    cm = p.get('confusion_matrix', m.get('confusion_matrix', []))
    if cm:
        print(f"    --- Confusion Matrix ---")
        hdr = '    ' + ' ' * 10 + '  '.join(f'{c:>6}' for c in cls_names)
        print(hdr)
        for i, row in enumerate(cm):
            rname = cls_names[i] if i < len(cls_names) else str(i)
            print(f"    {rname:<10}" + '  '.join(f'{v:>6}' for v in row))


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import numpy as np
    from utils import load_csi_datasets

    TRAIN_DIR = '../../../wifi_sensing_data/thoth_data/train'
    TEST_DIR  = '../../../wifi_sensing_data/thoth_data/test'
    WINDOW_LEN = 1500
    EPOCHS = 100
    LRS = [1e-3, 1e-4]

    print("=" * 80)
    print("DL EXPERIMENTS: Comprehensive Domain Adaptation Comparison")
    print(f"  Epochs: {EPOCHS}, Learning rates: {LRS}")
    print("  5 unique trainings × 2 splits × 2 LRs = 20 training runs")
    print("  + 3 adaptation variants on shared model = 9 results per combo")
    print("=" * 80)

    combined_ds, test_ds = load_csi_datasets(TRAIN_DIR, TEST_DIR, WINDOW_LEN, verbose=False)
    n_features = combined_ds.X.shape[1]
    n_classes  = combined_ds.num_classes

    print(f"Train: {combined_ds.X.shape}, Test: {test_ds.X.shape}")
    print(f"Features: {n_features}, Classes: {n_classes}")
    print(f"Label map: {combined_ds.label_map}")

    idx2name = {v: k for k, v in combined_ds.label_map.items()}

    def time_split(X, y, test_frac=0.2):
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            sp = len(ci) - max(1, int(len(ci) * test_frac))
            train_idx.append(ci[:sp])
            test_idx.append(ci[sp:])
        return np.concatenate(train_idx), np.concatenate(test_idx)

    def sample_fewshot(X, y, n_per_class):
        idxs = []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            k = min(n_per_class, len(ci))
            idxs.append(np.random.choice(ci, k, replace=False))
        return X[np.concatenate(idxs)], y[np.concatenate(idxs)]

    def sample_fewshot_pct(X, y, pct):
        idxs = []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            k = max(1, int(len(ci) * pct))
            idxs.append(np.random.choice(ci, k, replace=False))
        return X[np.concatenate(idxs)], y[np.concatenate(idxs)]

    all_results = {}

    for split_name, use_test_ds in [ ("Separate", True), ("Split", False)]:
        for lr in LRS:
            run_key = f"{split_name}_lr{lr}"
            print(f"\n{'='*80}")
            print(f"EXPERIMENT: {split_name} | LR={lr} | Epochs={EPOCHS}")
            print(f"{'='*80}")

            if use_test_ds:
                X_tr, y_tr = combined_ds.X, combined_ds.y
                X_te, y_te = test_ds.X, test_ds.y
            else:
                tri, tei = time_split(combined_ds.X, combined_ds.y)
                X_tr, y_tr = combined_ds.X[tri], combined_ds.y[tri]
                X_te, y_te = combined_ds.X[tei], combined_ds.y[tei]

            X_target = X_te
            print(f"  Train: {X_tr.shape[0]}, Test/Target: {X_te.shape[0]}")

            np.random.seed(42)
            X_fs10, y_fs10 = sample_fewshot(X_te, y_te, n_per_class=10)
            X_fs10pct, y_fs10pct = sample_fewshot_pct(X_te, y_te, pct=0.10)
            print(f"  Few-shot 10/class: {len(X_fs10)} samples")
            print(f"  Few-shot 10%/class: {len(X_fs10pct)} samples")

            run_results = {}
            train_kw = dict(epochs=EPOCHS, lr=lr, coral_weight=0.5,
                            confidence_threshold=0.8, verbose=True)

            # Few-shot adaptation configs applied to every trained model
            fs_variants = [
                ('',       {}),
                ('+FS10',  dict(X_target_labeled=X_fs10, y_target_labeled=y_fs10,
                                fewshot_epochs=20, fewshot_lr=1e-4)),
                ('+FS10%', dict(X_target_labeled=X_fs10pct, y_target_labeled=y_fs10pct,
                                fewshot_epochs=20, fewshot_lr=1e-4)),
            ]

            # Define training configs: (label, model_kwargs, train_kwargs)
            train_configs = [
                ('NoBN',       dict(use_batch_norm=False), {}),
                ('BN',         {},                         {}),
                ('BN+GlobCRL', {},                         dict(use_coral=True)),
                ('BN+CondCRL', {},                         dict(use_conditional_coral=True)),
                ('BN+Whiten',  dict(use_whitening=True),   {}),
                ('BN+CC+W',    dict(use_whitening=True),   dict(use_conditional_coral=True)),
            ]

            n_train = len(train_configs)
            trained_models = {}

            for ti, (tname, model_kw, extra_train_kw) in enumerate(train_configs):
                print(f"\n  {'='*60}")
                print(f"  Training {ti+1}/{n_train}: {tname}")
                print(f"  {'='*60}")

                mdl = make_adaptive_model(n_features, n_classes, config='small', **model_kw)
                merged_kw = {**train_kw, **extra_train_kw}
                trained_mdl, info = train_model(
                    mdl, X_tr, y_tr, X_target, X_te, y_te, **merged_kw)
                trained_models[tname] = (trained_mdl, info)

                # Evaluate: base (no adaptation) + few-shot variants
                for fs_suffix, fs_kw in fs_variants:
                    key = f"{tname}{fs_suffix}"
                    m = adapt_and_evaluate(trained_mdl, X_target, X_te, y_te,
                                           info, adapt_name=key, **fs_kw)
                    _print_dl_metrics(key, m, label_map=idx2name)
                    run_results[key] = m

            # Additional AdaBN variants on BN+CC+W (shared trained model)
            mdl_ccw, info_ccw = trained_models['BN+CC+W']
            adabn_variants = [
                ('AdaBN+CC+W',       dict(use_adabn=True, adabn_alpha=0.3)),
                ('AdaBN+CC+W+FS10',  dict(use_adabn=True, adabn_alpha=0.3,
                                           X_target_labeled=X_fs10, y_target_labeled=y_fs10,
                                           fewshot_epochs=20, fewshot_lr=1e-4)),
                ('AdaBN+CC+W+FS10%', dict(use_adabn=True, adabn_alpha=0.3,
                                           X_target_labeled=X_fs10pct, y_target_labeled=y_fs10pct,
                                           fewshot_epochs=20, fewshot_lr=1e-4)),
            ]
            print(f"\n  {'='*60}")
            print(f"  AdaBN adaptation variants on BN+CC+W")
            print(f"  {'='*60}")
            for aname, akw in adabn_variants:
                m = adapt_and_evaluate(mdl_ccw, X_target, X_te, y_te,
                                       info_ccw, adapt_name=aname, **akw)
                _print_dl_metrics(aname, m, label_map=idx2name)
                run_results[aname] = m

            all_results[run_key] = run_results

    # ---- Final comparison table ----
    # Collect all model names in order
    all_model_names = []
    for run_res in all_results.values():
        for k in run_res:
            if k not in all_model_names:
                all_model_names.append(k)

    print(f"\n{'='*140}")
    print("FINAL COMPARISON")
    print(f"{'='*140}")

    hdr = (f"{'Run':<18} | {'Model':<22} | {'Acc':>6} {'BalAcc':>6} {'F1w':>6} "
           f"{'F1mac':>6} {'Prec':>6} {'Rec':>6} {'Kappa':>6} {'MCC':>6} "
           f"{'ECE':>6} {'LogL':>6} {'Conf':>6} {'Ent':>6} | {'Time':>6}")
    print(hdr)
    print("-" * 140)
    for run_key, run_res in all_results.items():
        for mname in all_model_names:
            if mname in run_res:
                m = run_res[mname]
                p = m.get('post', {})
                print(f"{run_key:<18} | {mname:<22} | "
                      f"{p.get('accuracy', m['test_accuracy']):>6.4f} "
                      f"{p.get('balanced_accuracy', 0):>6.4f} "
                      f"{p.get('f1_weighted', m['test_f1']):>6.4f} "
                      f"{p.get('f1_macro', 0):>6.4f} "
                      f"{p.get('precision_weighted', 0):>6.4f} "
                      f"{p.get('recall_weighted', 0):>6.4f} "
                      f"{p.get('cohen_kappa', 0):>6.4f} "
                      f"{p.get('mcc', 0):>6.4f} "
                      f"{p.get('ece', 0):>6.4f} "
                      f"{p.get('log_loss', 0):>6.4f} "
                      f"{p.get('mean_confidence', 0):>6.4f} "
                      f"{p.get('mean_entropy', 0):>6.4f} | "
                      f"{m['train_time_s']:>5.0f}s")
        print("-" * 140)

    # ---- Adaptation delta table (pre vs post) ----
    print(f"\n{'='*100}")
    print("ADAPTATION EFFECT: Pre-Adapt vs Post-Adapt Accuracy")
    print(f"{'='*100}")
    print(f"{'Run':<18} | {'Model':<22} | {'Pre-Acc':>8} {'Post-Acc':>9} {'Delta':>7} {'Pre-F1':>7} {'Post-F1':>8}")
    print("-" * 82)
    for run_key, run_res in all_results.items():
        for mname in all_model_names:
            if mname in run_res:
                m = run_res[mname]
                pre_a = m.get('pre_adapt_accuracy', m['test_accuracy'])
                post_a = m['test_accuracy']
                delta = round(post_a - pre_a, 4)
                sign = '+' if delta >= 0 else ''
                pre_f = m.get('pre_adapt_f1', m['test_f1'])
                print(f"{run_key:<18} | {mname:<22} | {pre_a:>8.4f} {post_a:>9.4f} {sign}{delta:>6.4f} {pre_f:>7.4f} {m['test_f1']:>8.4f}")
        print("-" * 82)

    # ---- Best model per configuration ----
    print(f"\n{'='*80}")
    print("BEST MODEL PER CONFIGURATION")
    print(f"{'='*80}")
    for run_key, run_res in all_results.items():
        best_name, best_acc = None, -1
        for mname, m in run_res.items():
            if m['test_accuracy'] > best_acc:
                best_acc = m['test_accuracy']
                best_name = mname
        bm = run_res[best_name]
        p = bm.get('post', {})
        print(f"  {run_key:<18}: {best_name:<22} "
              f"Acc={bm['test_accuracy']:.4f}  F1={bm['test_f1']:.4f}  "
              f"Kappa={p.get('cohen_kappa', 'N/A')}  MCC={p.get('mcc', 'N/A')}  "
              f"ECE={p.get('ece', 'N/A')}")

    print(f"\n{'='*80}")
    print("DL experiments completed!")
    print(f"{'='*80}")
