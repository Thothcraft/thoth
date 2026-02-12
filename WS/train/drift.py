"""
Domain Drift Diagnostic Suite for CSI-based Activity Recognition.

Ten experiments to characterize the nature of distribution shift
between source (train) and target (test) domains:

 1. Domain Classifier Test        — Is there distribution shift at all?
 2. Per-Class Centroid Shift       — Is shift global or class-dependent?
 3. Covariance Structure Shift     — Is geometry distorted?
 4. Label Prior Shift              — Do class frequencies differ?
 5. Classifier Logit Shift         — Is there systematic logit bias?
 6. Linear Separability Test       — Are features good but boundary shifted?
 7. Maximum Mean Discrepancy       — Distribution distance beyond mean/cov
 8. Class-Conditional MMD          — Per-class distribution distance
 9. Subspace Angle Analysis        — PCA subspace rotation/scaling
10. Feature Whitening Test         — Is drift mostly second-order?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import cosine as cosine_dist
from scipy.linalg import sqrtm

from utils import load_csi_datasets
from dl import make_adaptive_model


# =============================================================================
# Helper: Train source model and extract features
# =============================================================================
def train_source_model(X_source, y_source, n_features, n_classes,
                       epochs=50, batch_size=64, lr=1e-3, verbose=True):
    """Train an AdaptiveModel on source data only. Returns trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_adaptive_model(n_features, n_classes, config='small')
    model = model.to(device)

    ds = TensorDataset(torch.FloatTensor(X_source), torch.LongTensor(y_source))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(logits, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
        if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{epochs} | Acc: {correct/total:.4f}")

    model.eval()
    return model


def extract_features(model, X, batch_size=256):
    """Extract features from a trained model."""
    device = next(model.parameters()).device
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            f = model.extract_features(xb)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_logits(model, X, batch_size=256):
    """Extract raw logits from a trained model."""
    device = next(model.parameters()).device
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            out = model(xb)
            logits_list.append(out.cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def _rbf_kernel(X, Y, sigma):
    """Compute RBF kernel matrix between X and Y."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2.0 * X @ Y.T
    return np.exp(-dists / (2.0 * sigma ** 2))


def _pairwise_sq_dists(X, Y):
    """Compute pairwise squared distances without 3D broadcast (memory-safe)."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    return XX + YY.T - 2.0 * X @ Y.T


def _compute_mmd(X, Y, sigma=None):
    """Compute MMD^2 with RBF kernel. Uses median heuristic for sigma if None."""
    if sigma is None:
        # Median heuristic on a subsample for efficiency
        max_med = min(200, len(X), len(Y))
        sub_X = X[np.random.choice(len(X), max_med, replace=False)] if len(X) > max_med else X
        sub_Y = Y[np.random.choice(len(Y), max_med, replace=False)] if len(Y) > max_med else Y
        sub = np.concatenate([sub_X, sub_Y], axis=0)
        dists = _pairwise_sq_dists(sub, sub)
        sigma = np.sqrt(np.median(dists[dists > 0]) / 2.0)
        if sigma < 1e-8:
            sigma = 1.0

    K_xx = _rbf_kernel(X, X, sigma)
    K_yy = _rbf_kernel(Y, Y, sigma)
    K_xy = _rbf_kernel(X, Y, sigma)

    n = len(X)
    m = len(Y)

    # Unbiased estimator
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)

    mmd2 = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2.0 * K_xy.mean()
    return float(mmd2), float(sigma)


# =============================================================================
# Experiment 1: Domain Classifier Test
# =============================================================================
def exp1_domain_classifier(feat_source, feat_target):
    """Domain classifier with 5-fold CV, AUC, and linear vs MLP comparison.

    Returns
    -------
    dict
        Accuracy/AUC for linear and MLP classifiers with confidence intervals.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Domain Classifier Test (5-fold CV)")
    print("=" * 70)
    print("  Goal: Can a classifier tell source from target features?")
    print("  If MLP >> Linear → nonlinear domain shift")

    X = np.concatenate([feat_source, feat_target], axis=0)
    y = np.concatenate([np.zeros(len(feat_source)), np.ones(len(feat_target))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    linear_accs, linear_aucs = [], []
    mlp_accs, mlp_aucs = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Linear
        clf_lin = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf_lin.fit(X_tr, y_tr)
        linear_accs.append(accuracy_score(y_te, clf_lin.predict(X_te)))
        linear_aucs.append(roc_auc_score(y_te, clf_lin.predict_proba(X_te)[:, 1]))

        # MLP
        clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                random_state=42, early_stopping=True)
        clf_mlp.fit(X_tr, y_tr)
        mlp_accs.append(accuracy_score(y_te, clf_mlp.predict(X_te)))
        mlp_aucs.append(roc_auc_score(y_te, clf_mlp.predict_proba(X_te)[:, 1]))

    lin_acc_m, lin_acc_s = np.mean(linear_accs), np.std(linear_accs)
    lin_auc_m, lin_auc_s = np.mean(linear_aucs), np.std(linear_aucs)
    mlp_acc_m, mlp_acc_s = np.mean(mlp_accs), np.std(mlp_accs)
    mlp_auc_m, mlp_auc_s = np.mean(mlp_aucs), np.std(mlp_aucs)

    print(f"\n  {'Classifier':<12} | {'Accuracy':>18} | {'ROC-AUC':>18}")
    print("  " + "-" * 54)
    print(f"  {'Linear':<12} | {lin_acc_m:.4f} +/- {lin_acc_s:.4f} | {lin_auc_m:.4f} +/- {lin_auc_s:.4f}")
    print(f"  {'MLP':<12} | {mlp_acc_m:.4f} +/- {mlp_acc_s:.4f} | {mlp_auc_m:.4f} +/- {mlp_auc_s:.4f}")

    gap = mlp_acc_m - lin_acc_m
    print(f"\n  MLP - Linear accuracy gap: {gap:+.4f}")

    if lin_acc_m < 0.55:
        print("  Interpretation: ~50% → No real distribution shift")
    elif lin_acc_m < 0.75:
        print("  Interpretation: Moderate covariate shift")
    else:
        print("  Interpretation: Strong domain shift")

    if gap > 0.05:
        print("  MLP >> Linear → Nonlinear domain shift detected")
    else:
        print("  MLP ~ Linear → Shift is mostly linear")

    return {
        'linear_acc': round(lin_acc_m, 4), 'linear_acc_std': round(lin_acc_s, 4),
        'linear_auc': round(lin_auc_m, 4), 'linear_auc_std': round(lin_auc_s, 4),
        'mlp_acc': round(mlp_acc_m, 4), 'mlp_acc_std': round(mlp_acc_s, 4),
        'mlp_auc': round(mlp_auc_m, 4), 'mlp_auc_std': round(mlp_auc_s, 4),
        'mlp_linear_gap': round(gap, 4),
    }


# =============================================================================
# Experiment 2: Per-Class Centroid Shift
# =============================================================================
def exp2_centroid_shift(feat_source, y_source, feat_target, y_target, label_map):
    """Compare per-class centroids: L2, Mahalanobis, and cosine similarity.

    Mahalanobis accounts for covariance geometry.
    Cosine similarity distinguishes scaling vs directional shift.

    Returns
    -------
    dict
        Per-class distances and global statistics.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Per-Class Centroid Shift (L2 + Mahalanobis + Cosine)")
    print("=" * 70)
    print("  Goal: Is shift global or class-dependent?")
    print("  Cosine high + L2 large → scaling shift")
    print("  Cosine low → directional shift")

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))

    # Compute pooled source covariance for Mahalanobis
    s_centered = feat_source - feat_source.mean(axis=0, keepdims=True)
    cov_s = (s_centered.T @ s_centered) / max(len(feat_source) - 1, 1)
    # Regularize for inversion
    reg = 1e-4 * np.eye(cov_s.shape[0])
    cov_s_inv = np.linalg.inv(cov_s + reg)

    rows = []
    for c in classes:
        s_mask = y_source == c
        t_mask = y_target == c
        if s_mask.sum() == 0 or t_mask.sum() == 0:
            continue

        mu_s = feat_source[s_mask].mean(axis=0)
        mu_t = feat_target[t_mask].mean(axis=0)
        diff = mu_s - mu_t

        l2 = float(np.linalg.norm(diff))
        mahal = float(np.sqrt(diff @ cov_s_inv @ diff))
        cos_sim = float(1.0 - cosine_dist(mu_s, mu_t))
        name = inv_map.get(c, str(c))
        rows.append({'name': name, 'l2': l2, 'mahal': mahal, 'cos': cos_sim})

    print(f"\n  {'Class':<12} | {'L2':>10} {'Mahalanobis':>12} {'Cosine Sim':>11}")
    print("  " + "-" * 50)
    for r in rows:
        print(f"  {r['name']:<12} | {r['l2']:>10.4f} {r['mahal']:>12.4f} {r['cos']:>11.4f}")

    l2s = [r['l2'] for r in rows]
    mahals = [r['mahal'] for r in rows]
    coss = [r['cos'] for r in rows]

    mean_l2, std_l2 = np.mean(l2s), np.std(l2s)
    cv_l2 = std_l2 / mean_l2 if mean_l2 > 0 else 0
    mean_mahal = np.mean(mahals)
    cv_mahal = np.std(mahals) / mean_mahal if mean_mahal > 0 else 0
    mean_cos = np.mean(coss)

    print(f"\n  L2:   mean={mean_l2:.4f}  std={std_l2:.4f}  CV={cv_l2:.4f}")
    print(f"  Mahal: mean={mean_mahal:.4f}  CV={cv_mahal:.4f}")
    print(f"  Cosine: mean={mean_cos:.4f}")

    if cv_l2 < 0.3:
        print("  Interpretation: All classes shift similarly → Global shift → AdaBN should help")
    elif cv_l2 < 0.7:
        print("  Interpretation: Moderate variation → Mixed shift")
    else:
        print("  Interpretation: High variation → Class-conditional shift → AdaBN may fail")

    if mean_cos > 0.9:
        print("  Cosine high → Mostly scaling/magnitude shift")
    elif mean_cos > 0.7:
        print("  Cosine moderate → Mixed scaling + directional shift")
    else:
        print("  Cosine low → Strong directional shift")

    return {
        'per_class': {r['name']: {'l2': round(r['l2'], 4), 'mahalanobis': round(r['mahal'], 4),
                                   'cosine': round(r['cos'], 4)} for r in rows},
        'mean_l2': round(mean_l2, 4), 'cv_l2': round(cv_l2, 4),
        'mean_mahalanobis': round(mean_mahal, 4), 'cv_mahalanobis': round(cv_mahal, 4),
        'mean_cosine': round(mean_cos, 4),
    }


# =============================================================================
# Experiment 3: Covariance Structure Shift
# =============================================================================
def exp3_covariance_shift(feat_source, feat_target):
    """Compare covariance matrices: relative Frobenius, principal angles, Grassmann distance.

    Returns
    -------
    dict
        Relative Frobenius, principal angles, Grassmann distance, eigenvalue comparison.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Covariance Structure Shift")
    print("=" * 70)
    print("  Goal: Is the feature geometry distorted?")

    d = feat_source.shape[1]

    # Center
    s_centered = feat_source - feat_source.mean(axis=0, keepdims=True)
    t_centered = feat_target - feat_target.mean(axis=0, keepdims=True)

    # Covariance matrices
    cov_s = (s_centered.T @ s_centered) / max(len(feat_source) - 1, 1)
    cov_t = (t_centered.T @ t_centered) / max(len(feat_target) - 1, 1)

    # Frobenius norms
    frob_diff = np.linalg.norm(cov_s - cov_t, 'fro')
    frob_s = np.linalg.norm(cov_s, 'fro')
    # Relative Frobenius: ||C_s - C_t||_F / ||C_s||_F
    frob_rel = frob_diff / max(frob_s, 1e-8)

    # Eigendecomposition (full)
    k = min(20, d)
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_s)
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_t)
    # Sort descending
    idx_s = np.argsort(eigvals_s)[::-1]
    idx_t = np.argsort(eigvals_t)[::-1]
    eigvals_s, eigvecs_s = eigvals_s[idx_s], eigvecs_s[:, idx_s]
    eigvals_t, eigvecs_t = eigvals_t[idx_t], eigvecs_t[:, idx_t]

    eig_s_top = eigvals_s[:k]
    eig_t_top = eigvals_t[:k]
    eig_ratio = eig_t_top / np.clip(eig_s_top, 1e-8, None)

    # Principal angles between top-k eigenspaces
    U_s = eigvecs_s[:, :k]  # (d, k)
    U_t = eigvecs_t[:, :k]  # (d, k)
    # SVD of U_s^T @ U_t gives cos(principal angles)
    cos_angles = np.linalg.svd(U_s.T @ U_t, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    principal_angles_deg = np.degrees(np.arccos(cos_angles))

    # Grassmann distance: sqrt(sum of squared principal angles in radians)
    principal_angles_rad = np.radians(principal_angles_deg)
    grassmann_dist = float(np.sqrt(np.sum(principal_angles_rad ** 2)))

    rank_corr = float(np.corrcoef(eig_s_top, eig_t_top)[0, 1])

    print(f"\n  Feature dimension:              {d}")
    print(f"  ||C_s - C_t||_F:                {frob_diff:.4f}")
    print(f"  ||C_s||_F:                      {frob_s:.4f}")
    print(f"  Relative Frobenius:             {frob_rel:.4f}")

    print(f"\n  Top-{k} eigenvalues:")
    print(f"  {'Rank':<6} | {'Source':>12} {'Target':>12} {'Ratio(T/S)':>12}")
    print("  " + "-" * 48)
    for i in range(k):
        print(f"  {i+1:<6} | {eig_s_top[i]:>12.4f} {eig_t_top[i]:>12.4f} {eig_ratio[i]:>12.4f}")

    print(f"\n  Eigenvalue rank correlation:    {rank_corr:.4f}")

    print(f"\n  Principal angles (top-{k} subspace):")
    print(f"  {'Angle #':<8} | {'Degrees':>10}")
    print("  " + "-" * 22)
    for i in range(min(k, 10)):
        print(f"  {i+1:<8} | {principal_angles_deg[i]:>10.2f}")
    if k > 10:
        print(f"  ... ({k - 10} more angles omitted)")

    mean_angle = float(np.mean(principal_angles_deg))
    max_angle = float(np.max(principal_angles_deg))
    print(f"\n  Mean principal angle:           {mean_angle:.2f} deg")
    print(f"  Max principal angle:            {max_angle:.2f} deg")
    print(f"  Grassmann distance:             {grassmann_dist:.4f}")

    if frob_rel < 0.1:
        print("  Interpretation: Small relative covariance difference")
    elif rank_corr > 0.9 and mean_angle < 15:
        print("  Interpretation: Covariance shifted but subspace preserved → CORAL should help")
    elif mean_angle > 30:
        print("  Interpretation: Significant subspace rotation → Geometric distortion")
    else:
        print("  Interpretation: Moderate geometric shift")

    return {
        'frobenius_diff': round(float(frob_diff), 4),
        'frobenius_relative': round(float(frob_rel), 4),
        'eigenvalue_rank_correlation': rank_corr,
        'mean_principal_angle_deg': round(mean_angle, 2),
        'max_principal_angle_deg': round(max_angle, 2),
        'grassmann_distance': round(grassmann_dist, 4),
        'top_eigenvalues_source': eig_s_top.tolist(),
        'top_eigenvalues_target': eig_t_top.tolist(),
    }


# =============================================================================
# Experiment 4: Label Prior Shift
# =============================================================================
def exp4_label_prior_shift(y_source, y_target, label_map):
    """Compare class frequency distributions P_s(y) vs P_t(y).

    Returns
    -------
    dict
        Per-class frequencies and KL divergence.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Label Prior Shift")
    print("=" * 70)
    print("  Goal: Do class frequencies differ between domains?")

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))

    p_s = np.array([(y_source == c).sum() for c in classes], dtype=float)
    p_t = np.array([(y_target == c).sum() for c in classes], dtype=float)

    p_s_norm = p_s / p_s.sum()
    p_t_norm = p_t / p_t.sum()

    # KL divergence: KL(P_t || P_s)
    kl = np.sum(p_t_norm * np.log(p_t_norm / np.clip(p_s_norm, 1e-8, None)))

    print(f"\n  {'Class':<12} | {'Source Count':>12} {'Source %':>9} | {'Target Count':>12} {'Target %':>9} | {'Ratio T/S':>10}")
    print("  " + "-" * 75)
    for i, c in enumerate(classes):
        name = inv_map.get(c, str(c))
        ratio = p_t_norm[i] / max(p_s_norm[i], 1e-8)
        print(f"  {name:<12} | {int(p_s[i]):>12} {p_s_norm[i]:>8.2%} | {int(p_t[i]):>12} {p_t_norm[i]:>8.2%} | {ratio:>10.4f}")

    print(f"\n  KL(P_target || P_source): {kl:.6f}")

    if kl < 0.01:
        print("  Interpretation: Negligible label prior shift")
    elif kl < 0.1:
        print("  Interpretation: Mild label prior shift")
    else:
        print("  Interpretation: Significant label prior shift → Entropy minimization may collapse to dominant class")

    return {
        'source_freq': p_s_norm.tolist(),
        'target_freq': p_t_norm.tolist(),
        'kl_divergence': float(kl),
    }


# =============================================================================
# Experiment 5: Classifier Logit Shift
# =============================================================================
def exp5_logit_shift(model, X_source, y_source, X_target, y_target, label_map):
    """Logit shift + calibration diagnostics: ECE, confidence histogram, entropy.

    Returns
    -------
    dict
        Per-class logit shifts, ECE, mean entropy, confidence stats.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Classifier Logit Shift + Calibration")
    print("=" * 70)
    print("  Goal: Systematic logit bias? Overconfident but wrong? Underconfident?")

    logits_s = extract_logits(model, X_source)
    logits_t = extract_logits(model, X_target)

    inv_map = {v: k for k, v in label_map.items()}
    n_classes = logits_s.shape[1]

    # Softmax
    def _softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    probs_s = _softmax(logits_s)
    probs_t = _softmax(logits_t)

    # ---- Logit shift table ----
    mean_s = logits_s.mean(axis=0)
    mean_t = logits_t.mean(axis=0)

    print(f"\n  Global mean logits per class:")
    print(f"  {'Class':<12} | {'Source':>10} {'Target':>10} {'Diff':>10}")
    print("  " + "-" * 46)
    for c in range(n_classes):
        name = inv_map.get(c, str(c))
        diff = mean_t[c] - mean_s[c]
        print(f"  {name:<12} | {mean_s[c]:>10.4f} {mean_t[c]:>10.4f} {diff:>+10.4f}")

    # Per-class logit for correct class
    print(f"\n  Per-class mean logits (samples of that class):")
    print(f"  {'Class':<12} | {'Src logit':>10} {'Tgt logit':>10} {'Shift':>10}")
    print("  " + "-" * 46)

    classes = sorted(np.unique(np.concatenate([y_source, y_target])))
    shifts = {}
    for c in classes:
        s_mask = y_source == c
        t_mask = y_target == c
        if s_mask.sum() == 0 or t_mask.sum() == 0:
            continue
        src_logit = logits_s[s_mask, c].mean()
        tgt_logit = logits_t[t_mask, c].mean()
        shift = tgt_logit - src_logit
        name = inv_map.get(c, str(c))
        shifts[name] = float(shift)
        print(f"  {name:<12} | {src_logit:>10.4f} {tgt_logit:>10.4f} {shift:>+10.4f}")

    # ---- Prediction distribution ----
    pred_t = np.argmax(logits_t, axis=1)
    pred_counts = np.bincount(pred_t, minlength=n_classes)
    dominant = inv_map.get(np.argmax(pred_counts), str(np.argmax(pred_counts)))
    dominant_pct = pred_counts.max() / pred_counts.sum()

    print(f"\n  Target predictions distribution:")
    for c in range(n_classes):
        name = inv_map.get(c, str(c))
        pct = pred_counts[c] / pred_counts.sum()
        bar = "#" * int(pct * 40)
        print(f"  {name:<12} | {pred_counts[c]:>6} ({pct:>6.1%}) {bar}")

    if dominant_pct > 0.5:
        print(f"\n  WARNING: {dominant} dominates ({dominant_pct:.1%}) → Possible 1-class collapse")

    # ---- Expected Calibration Error (ECE) ----
    n_bins = 10
    def _compute_ece(probs, labels, n_bins=10):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_data = []
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                bin_data.append((lo, hi, 0, 0, 0))
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(probs)) * abs(bin_acc - bin_conf)
            bin_data.append((lo, hi, bin_acc, bin_conf, bin_count))
        return ece, bin_data

    ece_s, bins_s = _compute_ece(probs_s, y_source)
    ece_t, bins_t = _compute_ece(probs_t, y_target)

    print(f"\n  Expected Calibration Error (ECE):")
    print(f"    Source ECE: {ece_s:.4f}")
    print(f"    Target ECE: {ece_t:.4f}")

    # ---- Confidence histogram ----
    conf_s = np.max(probs_s, axis=1)
    conf_t = np.max(probs_t, axis=1)

    print(f"\n  Confidence histogram (target):")
    print(f"  {'Bin':>12} | {'Count':>6} {'Accuracy':>10} {'Avg Conf':>10}")
    print("  " + "-" * 44)
    for lo, hi, acc, conf, cnt in bins_t:
        if cnt > 0:
            print(f"  ({lo:.1f}, {hi:.1f}]   | {int(cnt):>6} {acc:>10.4f} {conf:>10.4f}")
        else:
            print(f"  ({lo:.1f}, {hi:.1f}]   | {int(cnt):>6}        -          -")

    # ---- Mean softmax entropy ----
    def _entropy(p):
        return -np.sum(p * np.log(np.clip(p, 1e-8, 1.0)), axis=1)

    ent_s = _entropy(probs_s)
    ent_t = _entropy(probs_t)
    max_ent = np.log(n_classes)

    print(f"\n  Mean softmax entropy (max={max_ent:.4f}):")
    print(f"    Source: {ent_s.mean():.4f} (std={ent_s.std():.4f})")
    print(f"    Target: {ent_t.mean():.4f} (std={ent_t.std():.4f})")

    if ece_t > 0.2 and conf_t.mean() > 0.7:
        print("  Interpretation: Overconfident but wrong → Representation collapse likely")
    elif ece_t > 0.2 and conf_t.mean() < 0.5:
        print("  Interpretation: Underconfident → Boundary misalignment")
    elif ece_t < 0.1:
        print("  Interpretation: Well-calibrated on target")
    else:
        print("  Interpretation: Moderate miscalibration")

    return {
        'per_class_shifts': shifts,
        'ece_source': round(float(ece_s), 4),
        'ece_target': round(float(ece_t), 4),
        'mean_conf_source': round(float(conf_s.mean()), 4),
        'mean_conf_target': round(float(conf_t.mean()), 4),
        'mean_entropy_source': round(float(ent_s.mean()), 4),
        'mean_entropy_target': round(float(ent_t.mean()), 4),
    }


# =============================================================================
# Experiment 6: Linear Separability Test
# =============================================================================
def exp6_linear_separability(feat_source, y_source, feat_target, y_target, label_map,
                             n_repeats=10):
    """Test if labeled target data fixes performance. 10 repeats with mean±std.

    Returns
    -------
    dict
        Accuracy mean±std for each scenario and percentage.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 6: Linear Separability Test ({n_repeats} repeats)")
    print("=" * 70)
    print("  Goal: Are features good but boundary shifted?")
    print("  If curve saturates quickly → feature space is good")

    # Source only → target (deterministic, no variance)
    clf_src = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_src.fit(feat_source, y_source)
    acc_src_only = accuracy_score(y_target, clf_src.predict(feat_target))

    percentages = [0.05, 0.10, 0.20, 0.30, 0.50]
    results = {'source_only': {'mean': round(float(acc_src_only), 4), 'std': 0.0}}

    for pct in percentages:
        combined_accs = []
        tgt_only_accs = []

        for rep in range(n_repeats):
            rng = np.random.RandomState(rep)
            idx_labeled = []
            idx_test = []
            for c in np.unique(y_target):
                c_idx = np.where(y_target == c)[0]
                c_idx = rng.permutation(c_idx)
                n_c = max(1, int(len(c_idx) * pct))
                idx_labeled.extend(c_idx[:n_c].tolist())
                idx_test.extend(c_idx[n_c:].tolist())

            idx_labeled = np.array(idx_labeled)
            idx_test = np.array(idx_test)
            if len(idx_test) == 0:
                continue

            # Source + small target
            X_comb = np.concatenate([feat_source, feat_target[idx_labeled]])
            y_comb = np.concatenate([y_source, y_target[idx_labeled]])
            clf_c = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf_c.fit(X_comb, y_comb)
            combined_accs.append(accuracy_score(y_target[idx_test], clf_c.predict(feat_target[idx_test])))

            # Target only
            clf_t = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf_t.fit(feat_target[idx_labeled], y_target[idx_labeled])
            tgt_only_accs.append(accuracy_score(y_target[idx_test], clf_t.predict(feat_target[idx_test])))

        pct_key = int(pct * 100)
        results[f'source+{pct_key}%target'] = {
            'mean': round(float(np.mean(combined_accs)), 4),
            'std': round(float(np.std(combined_accs)), 4),
        }
        results[f'{pct_key}%target_only'] = {
            'mean': round(float(np.mean(tgt_only_accs)), 4),
            'std': round(float(np.std(tgt_only_accs)), 4),
        }

    print(f"\n  {'Scenario':<30} | {'Accuracy':>20}")
    print("  " + "-" * 54)
    for scenario, v in results.items():
        if isinstance(v, dict):
            print(f"  {scenario:<30} | {v['mean']:.4f} +/- {v['std']:.4f}")
        else:
            print(f"  {scenario:<30} | {v:>10.4f}")

    # Accuracy vs % target labels curve
    print(f"\n  Accuracy vs % target labels (source+target):")
    print(f"  {'%':>5} | {'Accuracy':>20} | {'Bar'}")
    print("  " + "-" * 50)
    print(f"  {'0':>5} | {acc_src_only:.4f}               | {'#' * int(acc_src_only * 40)}")
    for pct in percentages:
        pct_key = int(pct * 100)
        v = results[f'source+{pct_key}%target']
        bar = "#" * int(v['mean'] * 40)
        print(f"  {pct_key:>5} | {v['mean']:.4f} +/- {v['std']:.4f} | {bar}")

    gain_10 = results.get('source+10%target', {}).get('mean', acc_src_only) - acc_src_only
    if gain_10 > 0.05:
        print("\n  Interpretation: Small labeled target data helps significantly")
        print("  → Features are good, decision boundary shifted")
    else:
        print("\n  Interpretation: Adding target labels doesn't help much")
        print("  → Feature space itself may be distorted")

    # Check saturation
    gain_50 = results.get('source+50%target', {}).get('mean', acc_src_only) - acc_src_only
    if gain_50 > 0 and gain_10 / max(gain_50, 1e-8) > 0.5:
        print("  Curve saturates quickly → Feature space is usable")

    return results


# =============================================================================
# Experiment 7: Maximum Mean Discrepancy (MMD)
# =============================================================================
def exp7_mmd(feat_source, feat_target, X_source_raw, X_target_raw):
    """Compute MMD before and after feature extraction.

    If MMD increases after feature extraction → model amplifies drift.

    Returns
    -------
    dict
        MMD values before and after feature extraction.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Maximum Mean Discrepancy (MMD)")
    print("=" * 70)
    print("  Goal: Distribution distance beyond mean/covariance")
    print("  Compare MMD before vs after feature extractor")

    # Subsample for kernel computation — use smaller n for high-dim raw features
    max_n_raw = min(200, len(X_source_raw), len(X_target_raw))
    max_n_feat = min(500, len(feat_source), len(feat_target))

    if len(X_source_raw) > max_n_raw:
        idx_s = np.random.choice(len(X_source_raw), max_n_raw, replace=False)
        raw_s = X_source_raw[idx_s]
    else:
        raw_s = X_source_raw
    if len(X_target_raw) > max_n_raw:
        idx_t = np.random.choice(len(X_target_raw), max_n_raw, replace=False)
        raw_t = X_target_raw[idx_t]
    else:
        raw_t = X_target_raw

    print(f"  Computing MMD on raw input features (n={len(raw_s)}, d={raw_s.shape[1]})...")
    mmd_raw, sigma_raw = _compute_mmd(raw_s, raw_t)

    # Subsample extracted features
    if len(feat_source) > max_n_feat:
        idx_s = np.random.choice(len(feat_source), max_n_feat, replace=False)
        fs = feat_source[idx_s]
    else:
        fs = feat_source
    if len(feat_target) > max_n_feat:
        idx_t = np.random.choice(len(feat_target), max_n_feat, replace=False)
        ft = feat_target[idx_t]
    else:
        ft = feat_target

    print("  Computing MMD on extracted features...")
    mmd_feat, sigma_feat = _compute_mmd(fs, ft)

    print(f"\n  {'Stage':<25} | {'MMD^2':>12} {'Sigma':>10}")
    print("  " + "-" * 52)
    print(f"  {'Before feature extractor':<25} | {mmd_raw:>12.6f} {sigma_raw:>10.4f}")
    print(f"  {'After feature extractor':<25} | {mmd_feat:>12.6f} {sigma_feat:>10.4f}")

    if mmd_feat > mmd_raw * 1.5:
        print("\n  Interpretation: MMD INCREASED after feature extraction")
        print("  → Model AMPLIFIES domain drift — features are domain-specific")
    elif mmd_feat < mmd_raw * 0.5:
        print("\n  Interpretation: MMD decreased after feature extraction")
        print("  → Model partially aligns domains in feature space")
    else:
        print("\n  Interpretation: MMD similar before/after")
        print("  → Feature extractor neither amplifies nor reduces drift")

    return {
        'mmd_raw': round(float(mmd_raw), 6),
        'mmd_features': round(float(mmd_feat), 6),
        'sigma_raw': round(float(sigma_raw), 4),
        'sigma_features': round(float(sigma_feat), 4),
        'amplification_ratio': round(float(mmd_feat / max(mmd_raw, 1e-8)), 4),
    }


# =============================================================================
# Experiment 8: Class-Conditional MMD
# =============================================================================
def exp8_class_conditional_mmd(feat_source, y_source, feat_target, y_target, label_map):
    """Compute MMD per class for true class-conditional shift magnitude.

    Returns
    -------
    dict
        Per-class MMD values.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Class-Conditional MMD")
    print("=" * 70)
    print("  Goal: Per-class distribution distance (beyond centroids)")

    inv_map = {v: k for k, v in label_map.items()}
    classes = sorted(np.unique(np.concatenate([y_source, y_target])))

    per_class = {}
    for c in classes:
        s_mask = y_source == c
        t_mask = y_target == c
        if s_mask.sum() < 5 or t_mask.sum() < 5:
            continue

        fs = feat_source[s_mask]
        ft = feat_target[t_mask]

        # Subsample if needed
        max_n = 300
        if len(fs) > max_n:
            fs = fs[np.random.choice(len(fs), max_n, replace=False)]
        if len(ft) > max_n:
            ft = ft[np.random.choice(len(ft), max_n, replace=False)]

        mmd2, sigma = _compute_mmd(fs, ft)
        name = inv_map.get(c, str(c))
        per_class[name] = round(float(mmd2), 6)

    print(f"\n  {'Class':<12} | {'MMD^2':>12}")
    print("  " + "-" * 28)
    mmds = list(per_class.values())
    for name, m in per_class.items():
        print(f"  {name:<12} | {m:>12.6f}")

    mean_mmd = np.mean(mmds)
    std_mmd = np.std(mmds)
    cv_mmd = std_mmd / mean_mmd if mean_mmd > 0 else 0

    print(f"\n  Mean MMD^2:  {mean_mmd:.6f}")
    print(f"  Std MMD^2:   {std_mmd:.6f}")
    print(f"  CV:          {cv_mmd:.4f}")

    if cv_mmd < 0.3:
        print("  Interpretation: Uniform shift across classes → Global adaptation should work")
    elif cv_mmd < 0.7:
        print("  Interpretation: Moderate class-dependent shift")
    else:
        print("  Interpretation: Highly class-dependent shift → Need class-conditional adaptation")

    return {'per_class_mmd': per_class, 'mean': round(mean_mmd, 6),
            'std': round(std_mmd, 6), 'cv': round(cv_mmd, 4)}


# =============================================================================
# Experiment 9: Subspace Angle Analysis (PCA)
# =============================================================================
def exp9_subspace_angles(feat_source, feat_target):
    """PCA subspace analysis: principal angles distinguish rotation vs scaling drift.

    Returns
    -------
    dict
        Principal angles, explained variance ratios, rotation vs scaling diagnosis.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Subspace Angle Analysis (PCA)")
    print("=" * 70)
    print("  Goal: Rotation drift vs scaling drift?")
    print("  Large angles → rotation drift")
    print("  Small angles + eigenvalue change → scaling drift")

    d = feat_source.shape[1]
    k = min(20, d)

    # PCA on each domain
    s_centered = feat_source - feat_source.mean(axis=0, keepdims=True)
    t_centered = feat_target - feat_target.mean(axis=0, keepdims=True)

    U_s, S_s, Vt_s = np.linalg.svd(s_centered, full_matrices=False)
    U_t, S_t, Vt_t = np.linalg.svd(t_centered, full_matrices=False)

    # Explained variance ratios
    var_s = (S_s ** 2) / (S_s ** 2).sum()
    var_t = (S_t ** 2) / (S_t ** 2).sum()

    # Right singular vectors (principal directions)
    V_s = Vt_s[:k, :].T  # (d, k)
    V_t = Vt_t[:k, :].T  # (d, k)

    # Principal angles via SVD of V_s^T @ V_t
    cos_angles = np.linalg.svd(V_s.T @ V_t, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angles))

    # Eigenvalue ratio (squared singular values ~ eigenvalues of covariance)
    eig_s = (S_s[:k] ** 2) / max(len(feat_source) - 1, 1)
    eig_t = (S_t[:k] ** 2) / max(len(feat_target) - 1, 1)
    eig_ratio = eig_t / np.clip(eig_s, 1e-8, None)

    print(f"\n  Top-{k} PCA subspace analysis:")
    print(f"  {'PC':<5} | {'Angle(deg)':>10} {'VarRatio_S':>11} {'VarRatio_T':>11} {'EigRatio':>10}")
    print("  " + "-" * 52)
    for i in range(min(k, 15)):
        print(f"  {i+1:<5} | {angles_deg[i]:>10.2f} {var_s[i]:>11.4f} {var_t[i]:>11.4f} {eig_ratio[i]:>10.4f}")
    if k > 15:
        print(f"  ... ({k - 15} more components omitted)")

    mean_angle = float(np.mean(angles_deg))
    max_angle = float(np.max(angles_deg))
    mean_eig_ratio = float(np.mean(eig_ratio))
    std_eig_ratio = float(np.std(eig_ratio))

    # Cumulative explained variance
    cum_var_s = np.cumsum(var_s[:k])
    cum_var_t = np.cumsum(var_t[:k])

    print(f"\n  Mean principal angle:        {mean_angle:.2f} deg")
    print(f"  Max principal angle:         {max_angle:.2f} deg")
    print(f"  Mean eigenvalue ratio (T/S): {mean_eig_ratio:.4f} +/- {std_eig_ratio:.4f}")
    print(f"  Cumulative variance (k={k}):  Source={cum_var_s[-1]:.4f}  Target={cum_var_t[-1]:.4f}")

    # Diagnosis
    rotation = mean_angle > 20
    scaling = abs(mean_eig_ratio - 1.0) > 0.3 or std_eig_ratio > 0.5

    if rotation and scaling:
        print("  Interpretation: Both rotation AND scaling drift")
    elif rotation:
        print("  Interpretation: Rotation drift (subspace rotated, eigenvalues stable)")
    elif scaling:
        print("  Interpretation: Scaling drift (subspace stable, eigenvalues changed)")
    else:
        print("  Interpretation: Minimal subspace drift")

    return {
        'mean_angle_deg': round(mean_angle, 2),
        'max_angle_deg': round(max_angle, 2),
        'mean_eig_ratio': round(mean_eig_ratio, 4),
        'std_eig_ratio': round(std_eig_ratio, 4),
        'rotation_drift': rotation,
        'scaling_drift': scaling,
    }


# =============================================================================
# Experiment 10: Feature Whitening Test
# =============================================================================
def exp10_whitening_test(feat_source, y_source, feat_target, y_target, label_map):
    """Whiten both domains using source statistics, test if drift is second-order.

    Apply: x' = C_s^{-1/2} (x - mu_s)
    Then train classifier on whitened source, test on whitened target.

    If performance jumps → drift is mostly second-order (mean + covariance).
    If not → nonlinear drift.

    Returns
    -------
    dict
        Accuracy before and after whitening.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Feature Whitening Test")
    print("=" * 70)
    print("  Goal: Is drift mostly second-order (fixable by CORAL/AdaBN)?")

    # Baseline: source classifier on raw features
    clf_raw = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_raw.fit(feat_source, y_source)
    acc_raw = accuracy_score(y_target, clf_raw.predict(feat_target))

    # Compute source whitening transform
    mu_s = feat_source.mean(axis=0)
    s_centered = feat_source - mu_s
    cov_s = (s_centered.T @ s_centered) / max(len(feat_source) - 1, 1)

    # Regularized inverse square root
    reg = 1e-4 * np.eye(cov_s.shape[0])
    cov_s_reg = cov_s + reg

    # Use eigendecomposition for stable C^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(cov_s_reg)
    eigvals = np.clip(eigvals, 1e-6, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T  # Whitening matrix

    # Whiten source using source stats
    feat_source_w = (feat_source - mu_s) @ W.T

    # Whiten target using SOURCE stats (same transform)
    feat_target_w_src = (feat_target - mu_s) @ W.T

    # Whiten target using TARGET stats
    mu_t = feat_target.mean(axis=0)
    t_centered = feat_target - mu_t
    cov_t = (t_centered.T @ t_centered) / max(len(feat_target) - 1, 1)
    cov_t_reg = cov_t + reg
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_t_reg)
    eigvals_t = np.clip(eigvals_t, 1e-6, None)
    D_inv_sqrt_t = np.diag(1.0 / np.sqrt(eigvals_t))
    W_t = eigvecs_t @ D_inv_sqrt_t @ eigvecs_t.T
    feat_target_w_tgt = (feat_target - mu_t) @ W_t.T

    # Test 1: Source whitened → target whitened with source stats
    clf_w1 = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_w1.fit(feat_source_w, y_source)
    acc_w_src = accuracy_score(y_target, clf_w1.predict(feat_target_w_src))

    # Test 2: Source whitened → target whitened with own stats
    clf_w2 = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_w2.fit(feat_source_w, y_source)
    acc_w_tgt = accuracy_score(y_target, clf_w2.predict(feat_target_w_tgt))

    print(f"\n  {'Scenario':<45} | {'Accuracy':>10}")
    print("  " + "-" * 60)
    print(f"  {'Raw features (no whitening)':<45} | {acc_raw:>10.4f}")
    print(f"  {'Both whitened with source stats':<45} | {acc_w_src:>10.4f}")
    print(f"  {'Each whitened with own stats (oracle)':<45} | {acc_w_tgt:>10.4f}")

    gain_src = acc_w_src - acc_raw
    gain_tgt = acc_w_tgt - acc_raw

    print(f"\n  Gain (source whitening):  {gain_src:+.4f}")
    print(f"  Gain (oracle whitening):  {gain_tgt:+.4f}")

    if gain_tgt > 0.1:
        print("  Interpretation: Oracle whitening helps significantly")
        print("  → Drift IS mostly second-order → CORAL/AdaBN should help")
        if gain_src > 0.05:
            print("  Source whitening also helps → Shared whitening transform works")
        else:
            print("  But source whitening doesn't help → Need target-specific stats (AdaBN)")
    elif gain_tgt > 0.03:
        print("  Interpretation: Moderate second-order component")
        print("  → Partial benefit from CORAL/AdaBN, but nonlinear component exists")
    else:
        print("  Interpretation: Whitening doesn't help")
        print("  → Drift is NOT second-order → Nonlinear adaptation needed")

    return {
        'acc_raw': round(float(acc_raw), 4),
        'acc_whitened_source_stats': round(float(acc_w_src), 4),
        'acc_whitened_own_stats': round(float(acc_w_tgt), 4),
        'gain_source_whitening': round(float(gain_src), 4),
        'gain_oracle_whitening': round(float(gain_tgt), 4),
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    TRAIN_DIR = '../../../wifi_sensing_data/thoth_data/train'
    TEST_DIR  = '../../../wifi_sensing_data/thoth_data/test'
    WINDOW_LEN = 1500
    EPOCHS = 50

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("DOMAIN DRIFT DIAGNOSTIC SUITE (10 Experiments)")
    print("=" * 70)

    # ---- Load data ----
    print("\nLoading datasets...")
    train_ds, test_ds = load_csi_datasets(TRAIN_DIR, TEST_DIR, WINDOW_LEN, verbose=False)

    X_source, y_source = train_ds.X, train_ds.y
    X_target, y_target = test_ds.X, test_ds.y
    n_features = X_source.shape[1]
    n_classes = train_ds.num_classes
    label_map = train_ds.label_map

    print(f"  Source: {X_source.shape}, Target: {X_target.shape}")
    print(f"  Features: {n_features}, Classes: {n_classes}")
    print(f"  Label map: {label_map}")

    # ---- Train source model ----
    print("\n" + "=" * 70)
    print("Training source model...")
    print("=" * 70)
    model = train_source_model(X_source, y_source, n_features, n_classes,
                               epochs=EPOCHS, verbose=True)

    # ---- Extract features ----
    print("\nExtracting features...")
    feat_source = extract_features(model, X_source)
    feat_target = extract_features(model, X_target)
    print(f"  Source features: {feat_source.shape}")
    print(f"  Target features: {feat_target.shape}")

    # ---- Run all 10 experiments ----
    results = {}

    results['exp1'] = exp1_domain_classifier(feat_source, feat_target)

    results['exp2'] = exp2_centroid_shift(feat_source, y_source,
                                          feat_target, y_target, label_map)

    results['exp3'] = exp3_covariance_shift(feat_source, feat_target)

    results['exp4'] = exp4_label_prior_shift(y_source, y_target, label_map)

    results['exp5'] = exp5_logit_shift(model, X_source, y_source,
                                       X_target, y_target, label_map)

    results['exp6'] = exp6_linear_separability(feat_source, y_source,
                                               feat_target, y_target, label_map)

    results['exp7'] = exp7_mmd(feat_source, feat_target, X_source, X_target)

    results['exp8'] = exp8_class_conditional_mmd(feat_source, y_source,
                                                  feat_target, y_target, label_map)

    results['exp9'] = exp9_subspace_angles(feat_source, feat_target)

    results['exp10'] = exp10_whitening_test(feat_source, y_source,
                                             feat_target, y_target, label_map)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("DRIFT DIAGNOSTIC SUMMARY (10 Experiments)")
    print("=" * 70)

    # Exp1
    lin_acc = results['exp1']['linear_acc']
    mlp_acc = results['exp1']['mlp_acc']
    gap = results['exp1']['mlp_linear_gap']
    print(f"\n  1. Domain classifier (linear): {lin_acc:.4f}  ", end="")
    print("(strong shift)" if lin_acc > 0.75 else "(moderate)" if lin_acc > 0.55 else "(no shift)")
    print(f"     Domain classifier (MLP):    {mlp_acc:.4f}  gap={gap:+.4f}  ", end="")
    print("(nonlinear shift)" if gap > 0.05 else "(linear shift)")

    # Exp2
    cv = results['exp2']['cv_l2']
    cos = results['exp2']['mean_cosine']
    print(f"  2. Centroid CV(L2):            {cv:.4f}  ", end="")
    print("(class-conditional)" if cv > 0.7 else "(mixed)" if cv > 0.3 else "(global)")
    print(f"     Mean cosine similarity:     {cos:.4f}  ", end="")
    print("(scaling)" if cos > 0.9 else "(mixed)" if cos > 0.7 else "(directional)")

    # Exp3
    frob_rel = results['exp3']['frobenius_relative']
    mean_angle = results['exp3']['mean_principal_angle_deg']
    grassmann = results['exp3']['grassmann_distance']
    print(f"  3. Relative Frobenius:         {frob_rel:.4f}")
    print(f"     Mean principal angle:       {mean_angle:.2f} deg")
    print(f"     Grassmann distance:         {grassmann:.4f}")

    # Exp4
    kl = results['exp4']['kl_divergence']
    print(f"  4. Label prior KL:             {kl:.6f}  ", end="")
    print("(significant)" if kl > 0.1 else "(mild)" if kl > 0.01 else "(negligible)")

    # Exp5
    ece_t = results['exp5']['ece_target']
    ent_t = results['exp5']['mean_entropy_target']
    conf_t = results['exp5']['mean_conf_target']
    print(f"  5. Target ECE:                 {ece_t:.4f}")
    print(f"     Target mean confidence:     {conf_t:.4f}")
    print(f"     Target mean entropy:        {ent_t:.4f}")

    # Exp6
    src_acc = results['exp6']['source_only']['mean']
    adapted_acc = results['exp6'].get('source+10%target', {}).get('mean', src_acc)
    print(f"  6. Source-only linear acc:     {src_acc:.4f}")
    print(f"     +10% target linear acc:     {adapted_acc:.4f}  ", end="")
    gain = adapted_acc - src_acc
    print(f"(+{gain:.4f})" if gain > 0.05 else "(minimal)")

    # Exp7
    mmd_raw = results['exp7']['mmd_raw']
    mmd_feat = results['exp7']['mmd_features']
    amp = results['exp7']['amplification_ratio']
    print(f"  7. MMD raw:                    {mmd_raw:.6f}")
    print(f"     MMD features:               {mmd_feat:.6f}  ratio={amp:.2f}x  ", end="")
    print("(amplified)" if amp > 1.5 else "(reduced)" if amp < 0.5 else "(stable)")

    # Exp8
    cv_mmd = results['exp8']['cv']
    print(f"  8. Class-conditional MMD CV:   {cv_mmd:.4f}  ", end="")
    print("(class-dependent)" if cv_mmd > 0.7 else "(mixed)" if cv_mmd > 0.3 else "(uniform)")

    # Exp9
    rot = results['exp9']['rotation_drift']
    scl = results['exp9']['scaling_drift']
    print(f"  9. Subspace: rotation={rot}, scaling={scl}  ", end="")
    if rot and scl:
        print("(both)")
    elif rot:
        print("(rotation only)")
    elif scl:
        print("(scaling only)")
    else:
        print("(minimal)")

    # Exp10
    acc_raw = results['exp10']['acc_raw']
    gain_oracle = results['exp10']['gain_oracle_whitening']
    print(f" 10. Whitening raw acc:          {acc_raw:.4f}")
    print(f"     Oracle whitening gain:      {gain_oracle:+.4f}  ", end="")
    print("(second-order)" if gain_oracle > 0.1 else "(partial)" if gain_oracle > 0.03 else "(nonlinear)")

    # ---- Final diagnosis ----
    print(f"\n{'='*70}")
    print("FINAL DIAGNOSIS")
    print(f"{'='*70}")

    shift_types = []
    if lin_acc > 0.75:
        shift_types.append("Strong covariate shift")
    if gap > 0.05:
        shift_types.append("Nonlinear component in shift")
    if cv > 0.7:
        shift_types.append("Class-conditional shift")
    elif cv < 0.3:
        shift_types.append("Global (class-uniform) shift")
    if mean_angle > 30:
        shift_types.append("Subspace rotation")
    if results['exp9']['scaling_drift']:
        shift_types.append("Eigenvalue scaling drift")
    if gain_oracle > 0.1:
        shift_types.append("Drift is mostly second-order")
    elif gain_oracle < 0.03:
        shift_types.append("Drift has strong nonlinear component")
    if gain > 0.05:
        shift_types.append("Decision boundary shifted (features usable)")

    for i, s in enumerate(shift_types):
        print(f"  {i+1}. {s}")

    print(f"\n  Recommended adaptation strategy:")
    if gain_oracle > 0.1 and cv < 0.5:
        print("  → CORAL + AdaBN should be effective (second-order, global shift)")
    elif gain > 0.05 and gain_oracle < 0.05:
        print("  → Fine-tune classifier head with small labeled target data")
    elif gap > 0.05:
        print("  → Need nonlinear adaptation (adversarial or deeper methods)")
    else:
        print("  → Try CORAL + AdaBN + TTA; consider few-shot target fine-tuning")

    print(f"\n{'='*70}")
    print("Drift diagnostics completed!")
    print(f"{'='*70}")
