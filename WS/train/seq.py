"""
Sequence Encoder for CSI-based Human Activity Recognition.

Architecture
------------
    Input (B, T*C flattened) -> reshape (B, C, T)
        -> Conv1D blocks (BN + ReLU + Pool)
        -> Bidirectional LSTM
        -> Temporal pooling  ->  z  (shared representation)

Heads (trained simultaneously):
    z -> HAR classifier          (cross-entropy)
    z -> Reconstruction decoder  (MSE, reverses encoder)
    z -> Contrastive projection  (NT-Xent on augmented pairs)
    z -> Domain discriminator    (GRL, binary CE: train vs train_2)

Regularisation:
    - BatchNorm throughout Conv1D + LSTM output
    - FeatureWhitening (ZCA-style, learnable decorrelation)
    - Conditional CORAL (per-class covariance alignment across domains)
"""

import math
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Gradient Reversal Layer
# =============================================================================
class GradientReversalFn(torch.autograd.Function):
    """Reverses gradients during backward pass, scaled by *lambda*."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps :class:`GradientReversalFn` as an ``nn.Module``.

    Parameters
    ----------
    lam : float
        Gradient scaling factor (negated in backward).  Typically annealed
        from 0 → 1 during training via :meth:`set_lambda`.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def set_lambda(self, lam: float):
        self.lam = lam

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lam)


# =============================================================================
# Feature Whitening  (re-used from dl.py convention)
# =============================================================================
class FeatureWhitening(nn.Module):
    """BN (per-dim normalisation) + learnable decorrelation (rotation).

    Parameters
    ----------
    num_features : int
        Feature dimension.
    momentum : float
        BN momentum.  Default 0.1.
    """

    def __init__(self, num_features: int, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False, momentum=momentum)
        self.decorrelate = nn.Linear(num_features, num_features, bias=False)
        nn.init.eye_(self.decorrelate.weight)

    def forward(self, x):
        return self.decorrelate(self.bn(x))


# =============================================================================
# Conv1D–LSTM Encoder
# =============================================================================
class Conv1dBlock(nn.Module):
    """Conv1D -> BN -> ReLU -> MaxPool."""

    def __init__(self, in_ch, out_ch, kernel_size=5, pool_size=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class SeqEncoder(nn.Module):
    """Conv1D + Bi-LSTM encoder producing a fixed-length representation.

    Input  : (B, input_dim)          – flattened window (T*C)
    Output : (B, repr_dim)           – pooled hidden state

    Parameters
    ----------
    n_subcarriers : int   Number of CSI subcarriers (channels).
    window_len    : int   Temporal length of one window.
    conv_channels : list  Channel sizes for successive Conv1D blocks.
    lstm_hidden   : int   LSTM hidden size (per direction).
    lstm_layers   : int   Number of stacked LSTM layers.
    dropout       : float Dropout in LSTM + between Conv blocks.
    use_whitening : bool  Append FeatureWhitening after pooling.
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        window_len: int = 2000,
        conv_channels: list | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_whitening: bool = True,
    ):
        super().__init__()
        self.n_subcarriers = n_subcarriers
        self.window_len = window_len
        if conv_channels is None:
            conv_channels = [64, 128]

        # --- Conv1D stack (channels-first: B, C_in, T) ---
        layers = []
        in_ch = n_subcarriers
        for out_ch in conv_channels:
            layers.append(Conv1dBlock(in_ch, out_ch))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)

        # Compute temporal length after conv pooling
        t = window_len
        for _ in conv_channels:
            t = t // 2  # MaxPool1d(2) halves each time
        self._conv_time = t
        self._conv_out_ch = conv_channels[-1]

        # --- Bi-LSTM ---
        self.lstm = nn.LSTM(
            input_size=self._conv_out_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.repr_dim = lstm_hidden * 2  # bidirectional
        self.pool_norm = nn.BatchNorm1d(self.repr_dim)

        # --- Optional whitening ---
        self.use_whitening = use_whitening
        if use_whitening:
            self.whitening = FeatureWhitening(self.repr_dim)

    @property
    def output_dim(self) -> int:
        return self.repr_dim

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, input_dim)  flattened CSI window.

        Returns
        -------
        z : (B, repr_dim)   pooled representation.
        """
        B = x.size(0)
        # reshape to (B, n_subcarriers, window_len)
        x = x.view(B, self.window_len, self.n_subcarriers).permute(0, 2, 1)

        # Conv1D stack: (B, C_out, T')
        x = self.convs(x)

        # Prepare for LSTM: (B, T', C_out)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # (B, T', 2*H)

        # Temporal mean-pool -> (B, 2*H)
        z = x.mean(dim=1)
        z = self.pool_norm(z)

        if self.use_whitening:
            z = self.whitening(z)
        return z


# =============================================================================
# Task Heads
# =============================================================================
class HARClassifier(nn.Module):
    """Activity classification head."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class ReconstructionDecoder(nn.Module):
    """Mirrors encoder: Linear -> un-LSTM (GRU) -> TransposedConv1D -> flatten.

    A lightweight decoder sufficient for a reconstruction auxiliary signal;
    not aiming for perfect inversion.

    Parameters
    ----------
    repr_dim      : int   Encoder output dimension.
    n_subcarriers : int   Target subcarrier count (output channels).
    window_len    : int   Target temporal length.
    conv_time     : int   Temporal length entering the LSTM in the encoder.
    conv_out_ch   : int   Channel count entering the LSTM in the encoder.
    """

    def __init__(self, repr_dim: int, n_subcarriers: int, window_len: int,
                 conv_time: int, conv_out_ch: int):
        super().__init__()
        self.conv_time = conv_time
        self.conv_out_ch = conv_out_ch
        self.window_len = window_len
        self.n_subcarriers = n_subcarriers

        self.fc = nn.Linear(repr_dim, conv_time * conv_out_ch)
        self.gru = nn.GRU(conv_out_ch, conv_out_ch, batch_first=True)
        self.upsample = nn.Upsample(size=window_len, mode='linear', align_corners=False)
        self.proj = nn.Conv1d(conv_out_ch, n_subcarriers, kernel_size=1)

    def forward(self, z):
        """
        Parameters
        ----------
        z : (B, repr_dim)

        Returns
        -------
        x_hat : (B, input_dim)  flattened reconstruction.
        """
        B = z.size(0)
        h = self.fc(z).view(B, self.conv_time, self.conv_out_ch)
        h, _ = self.gru(h)                         # (B, T', C)
        h = h.permute(0, 2, 1)                     # (B, C, T')
        h = self.upsample(h)                        # (B, C, window_len)
        h = self.proj(h)                            # (B, n_sub, window_len)
        return h.permute(0, 2, 1).reshape(B, -1)   # (B, window_len * n_sub)


class ContrastiveHead(nn.Module):
    """MLP projection for SimCLR-style NT-Xent contrastive learning."""

    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class DomainDiscriminator(nn.Module):
    """Binary domain classifier behind a Gradient Reversal Layer."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.grl = GradientReversalLayer(lam=1.0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def set_lambda(self, lam: float):
        self.grl.set_lambda(lam)

    def forward(self, z):
        return self.net(self.grl(z)).squeeze(-1)


# =============================================================================
# Loss helpers
# =============================================================================
def nt_xent_loss(z_i, z_j, temperature: float = 0.5):
    """Normalised Temperature-scaled Cross-Entropy (SimCLR).

    Parameters
    ----------
    z_i, z_j : (B, D)  L2-normalised projections of two augmented views.
    temperature : float  Scaling temperature.

    Returns
    -------
    Scalar loss.
    """
    B = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)                   # (2B, D)
    sim = z @ z.t() / temperature                       # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_select(mask).view(2 * B, 2 * B - 1)

    # Positive pair indices: i <-> i+B (shifted by diagonal removal)
    pos_idx = torch.arange(B, device=z.device)
    # For row i      (first half): positive is at column i+B-1   (−1 for diag removal)
    # For row i+B (second half): positive is at column i         (no shift needed since diag at i+B removed)
    labels_top = pos_idx + B - 1
    labels_bot = pos_idx
    labels = torch.cat([labels_top, labels_bot], dim=0)

    return F.cross_entropy(sim, labels)


def conditional_coral_loss(source_feats, target_feats, source_labels,
                           target_logits, confidence_threshold=0.8):
    """Per-class covariance alignment via pseudo-labels (see dl.py)."""
    d = source_feats.size(1)

    with torch.no_grad():
        probs = F.softmax(target_logits, dim=1)
        max_probs, pseudo = probs.max(dim=1)
        conf_mask = max_probs >= confidence_threshold

    total = torch.tensor(0.0, device=source_feats.device)
    n = 0
    for c in source_labels.unique():
        s = source_feats[source_labels == c]
        t = target_feats[(pseudo == c) & conf_mask]
        if s.size(0) < 2 or t.size(0) < 2:
            continue
        cov_s = _cov(s)
        cov_t = _cov(t)
        total = total + (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
        n += 1
    return total / max(n, 1)


def _cov(x):
    x = x - x.mean(0, keepdim=True)
    return (x.t() @ x) / max(x.size(0) - 1, 1)


# =============================================================================
# Simple augmentation for contrastive pairs
# =============================================================================
def augment_batch(x, noise_std=0.05, scale_range=(0.9, 1.1)):
    """Gaussian noise + random amplitude scaling (operates on flat vectors)."""
    noise = torch.randn_like(x) * noise_std
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(*scale_range)
    return x * scale + noise


# =============================================================================
# Full Model
# =============================================================================
class SeqModel(nn.Module):
    """Multi-task sequence model combining all components.

    Attributes
    ----------
    encoder        : SeqEncoder
    classifier     : HARClassifier
    decoder        : ReconstructionDecoder
    contrastive    : ContrastiveHead
    discriminator  : DomainDiscriminator
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        window_len: int = 2000,
        num_classes: int = 7,
        conv_channels: list | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_whitening: bool = True,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.encoder = SeqEncoder(
            n_subcarriers=n_subcarriers,
            window_len=window_len,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            use_whitening=use_whitening,
        )
        rdim = self.encoder.output_dim

        self.classifier = HARClassifier(rdim, num_classes, dropout=dropout)
        self.decoder = ReconstructionDecoder(
            repr_dim=rdim,
            n_subcarriers=n_subcarriers,
            window_len=window_len,
            conv_time=self.encoder._conv_time,
            conv_out_ch=self.encoder._conv_out_ch,
        )
        self.contrastive = ContrastiveHead(rdim, proj_dim)
        self.discriminator = DomainDiscriminator(rdim)

        # Alias for API compat with dl.py helpers
        self.label_classifier = self.classifier
        self.num_classes = num_classes

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --- forward helpers ---
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def extract_features(self, x):
        return self.encoder(x)

    def predict(self, x, batch_size=256):
        if x.size(0) <= batch_size:
            return self.forward(x)
        parts = [self.forward(x[i:i+batch_size]) for i in range(0, x.size(0), batch_size)]
        return torch.cat(parts, 0)


# =============================================================================
# GRL lambda schedule  (sigmoidal ramp from 0 → 1)
# =============================================================================
def grl_lambda_schedule(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    p = epoch / max(total_epochs - 1, 1)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


# =============================================================================
# Training loop
# =============================================================================
def train_seq_model(
    model: SeqModel,
    X_src: np.ndarray,   y_src: np.ndarray,
    X_tgt: np.ndarray,   y_tgt: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    X_calib: np.ndarray | None = None,
    *,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    # Loss weights
    w_cls: float = 1.0,
    w_recon: float = 0.5,
    w_contrast: float = 0.3,
    w_domain: float = 0.3,
    w_coral: float = 0.3,
    # Conditional CORAL
    use_cond_coral: bool = True,
    coral_conf: float = 0.8,
    # Contrastive
    temperature: float = 0.5,
    noise_std: float = 0.05,
    verbose: bool = True,
):
    """Train :class:`SeqModel` with all auxiliary objectives.

    Parameters
    ----------
    X_src, y_src : Source domain (train)   – numpy arrays.
    X_tgt, y_tgt : Target domain (train_2) – numpy arrays.
    X_test, y_test : Held-out test set.
    X_calib : Calibration data for reconstruction (unlabelled). Falls back to
        source data if None.
    epochs, batch_size, lr : standard hyper-parameters.
    w_cls, w_recon, w_contrast, w_domain, w_coral : loss weights.
    use_cond_coral : enable conditional CORAL between domains.
    coral_conf : pseudo-label confidence threshold for CondCORAL.
    temperature : NT-Xent temperature.
    noise_std : augmentation noise for contrastive pairs.
    verbose : print epoch-level metrics.

    Returns
    -------
    model   : trained SeqModel (eval mode, on device).
    info    : dict with training metadata.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Data loaders
    src_ds = TensorDataset(torch.FloatTensor(X_src), torch.LongTensor(y_src))
    tgt_ds = TensorDataset(torch.FloatTensor(X_tgt), torch.LongTensor(y_tgt))
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Calibration loader for reconstruction (falls back to source if not provided)
    calib_tensor = torch.FloatTensor(X_calib if X_calib is not None else X_src)
    calib_loader = DataLoader(
        TensorDataset(calib_tensor), batch_size=batch_size, shuffle=True, drop_last=True)

    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    criterion_domain = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    report_every = max(1, epochs // 10)
    t0 = time.time()

    if verbose:
        print(f"  Device: {device}  |  Epochs: {epochs}  |  LR: {lr}")
        print(f"  Weights  cls={w_cls}  recon={w_recon}  contrast={w_contrast}  "
              f"domain={w_domain}  coral={w_coral}")
        print(f"  CondCORAL: {use_cond_coral} (conf={coral_conf})")
        print(f"  Reconstruction on: {'calibration data' if X_calib is not None else 'source data'}")

    for epoch in range(epochs):
        model.train()

        # Anneal GRL lambda 0 → 1
        lam = grl_lambda_schedule(epoch, epochs)
        model.discriminator.set_lambda(lam)

        run = {k: 0.0 for k in ['cls', 'recon', 'contr', 'dom', 'coral', 'total']}
        correct, total = 0, 0

        tgt_iter = iter(tgt_loader)
        cal_iter = iter(calib_loader)

        for xb_s, yb_s in src_loader:
            # Fetch a target batch (cycle if shorter)
            try:
                xb_t, yb_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xb_t, yb_t = next(tgt_iter)

            # Fetch a calibration batch (cycle if shorter)
            try:
                (xb_c,) = next(cal_iter)
            except StopIteration:
                cal_iter = iter(calib_loader)
                (xb_c,) = next(cal_iter)
            xb_c = xb_c.to(device)

            xb_s, yb_s = xb_s.to(device), yb_s.to(device)
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            B_s, B_t = xb_s.size(0), xb_t.size(0)

            # --- Encode both domains ---
            z_s = model.encode(xb_s)
            z_t = model.encode(xb_t)

            # 1) HAR classification (source + target — both labelled)
            logits_s = model.classifier(z_s)
            logits_t = model.classifier(z_t)
            loss_cls = criterion_cls(logits_s, yb_s) + criterion_cls(logits_t, yb_t)

            # 2) Reconstruction (calibration data)
            z_c = model.encode(xb_c)
            x_hat_c = model.decoder(z_c)
            loss_recon = criterion_recon(x_hat_c, xb_c)

            # 3) Contrastive (augmented views of combined batch)
            xb_all = torch.cat([xb_s, xb_t], dim=0)
            z_all = torch.cat([z_s, z_t], dim=0)
            xb_aug = augment_batch(xb_all, noise_std=noise_std)
            z_aug = model.encode(xb_aug)
            proj_orig = model.contrastive(z_all)
            proj_aug = model.contrastive(z_aug)
            loss_contr = nt_xent_loss(proj_orig, proj_aug, temperature)

            # 4) Domain discrimination via GRL (source=0, target=1)
            d_s = model.discriminator(z_s)
            d_t = model.discriminator(z_t)
            dom_labels = torch.cat([
                torch.zeros(B_s, device=device),
                torch.ones(B_t, device=device),
            ])
            loss_dom = criterion_domain(torch.cat([d_s, d_t]), dom_labels)

            # 5) Conditional CORAL
            loss_coral = torch.tensor(0.0, device=device)
            if use_cond_coral:
                loss_coral = conditional_coral_loss(
                    z_s, z_t.detach(), yb_s, logits_t.detach(), coral_conf)

            # --- Combined loss ---
            loss = (w_cls * loss_cls
                    + w_recon * loss_recon
                    + w_contrast * loss_contr
                    + w_domain * loss_dom
                    + w_coral * loss_coral)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Bookkeeping
            n = B_s + B_t
            run['cls']    += loss_cls.item() * n
            run['recon']  += loss_recon.item() * n
            run['contr']  += loss_contr.item() * n
            run['dom']    += loss_dom.item() * n
            run['coral']  += loss_coral.item() * n
            run['total']  += loss.item() * n
            total += n

            _, preds_s = logits_s.max(1)
            _, preds_t = logits_t.max(1)
            correct += (preds_s == yb_s).sum().item() + (preds_t == yb_t).sum().item()

        scheduler.step()
        train_acc = correct / max(total, 1)

        # --- Periodic evaluation ---
        if verbose and ((epoch + 1) % report_every == 0 or epoch == 0):
            model.eval()
            with torch.no_grad():
                test_logits = model.predict(X_test_t)
                test_loss = criterion_cls(test_logits, y_test_t).item()
                test_acc = (test_logits.argmax(1) == y_test_t).float().mean().item()

            avg = {k: v / max(total, 1) for k, v in run.items()}
            print(
                f"  Ep {epoch+1:3d}/{epochs} | "
                f"Loss {avg['total']:.4f} "
                f"(cls={avg['cls']:.3f} rec={avg['recon']:.3f} "
                f"ctr={avg['contr']:.3f} dom={avg['dom']:.3f} "
                f"cor={avg['coral']:.4f}) | "
                f"TrainAcc {train_acc:.4f} | "
                f"TestLoss {test_loss:.4f}  TestAcc {test_acc:.4f} | "
                f"λ_grl={lam:.3f}"
            )

    train_time = round(time.time() - t0, 2)
    model.eval()

    if verbose:
        print(f"  Training complete in {train_time}s  (final train acc: {train_acc:.4f})")

    return model, {
        'train_time_s': train_time,
        'train_accuracy': round(train_acc, 4),
        'epochs': epochs,
        'lr': lr,
    }


# =============================================================================
# Compute metrics  (light wrapper matching dl.py interface)
# =============================================================================
def compute_metrics(model, X_test, y_test, device=None):
    """Evaluate model and return comprehensive metrics dict."""
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        precision_score, recall_score,
        cohen_kappa_score, matthews_corrcoef,
        balanced_accuracy_score, log_loss,
    )

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        logits = model.predict(torch.FloatTensor(X_test).to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    n = len(y_test)
    n_cls = logits.size(1)

    acc     = round(accuracy_score(y_test, preds), 4)
    bal_acc = round(balanced_accuracy_score(y_test, preds), 4)
    f1_w    = round(f1_score(y_test, preds, average='weighted', zero_division=0), 4)
    f1_mac  = round(f1_score(y_test, preds, average='macro',    zero_division=0), 4)
    kappa   = round(cohen_kappa_score(y_test, preds), 4)
    mcc     = round(matthews_corrcoef(y_test, preds), 4)

    max_p = probs.max(axis=1)
    ent   = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)

    # ECE (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        m = (max_p > bins[b]) & (max_p <= bins[b + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs((preds[m] == y_test[m]).mean() - max_p[m].mean())
    ece = round(float(ece), 4)

    try:
        ll = round(log_loss(y_test, probs, labels=list(range(n_cls))), 4)
    except Exception:
        ll = float('nan')

    cm = confusion_matrix(y_test, preds, labels=list(range(n_cls)))

    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'f1_weighted': f1_w, 'f1_macro': f1_mac,
        'cohen_kappa': kappa, 'mcc': mcc,
        'ece': ece, 'log_loss': ll,
        'mean_confidence': round(float(max_p.mean()), 4),
        'mean_entropy':    round(float(ent.mean()), 4),
        'confusion_matrix': cm.tolist(),
    }


# =============================================================================
# Main: experiments matching dl.py conventions
# =============================================================================
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import glob
    from utils import (load_csi_datasets, Pipeline, CSI_Loader,
                       FeatureSelector, WindowTransformer)

    # ----- Paths & hyper-parameters -----
    TRAIN_DIR  = '../../../wifi_sensing_data/har_data/train'
    TRAIN_DIR2 = '../../../wifi_sensing_data/har_data/train_2'
    TEST_DIR   = '../../../wifi_sensing_data/har_data/test'
    CALIBRATION_DIR   = '../../../wifi_sensing_data/calibration_data'
    WINDOW_LEN = 1000
    EPOCHS     = 800
    LR         = 1e-3

    print("=" * 80)
    print("SEQ EXPERIMENTS: Conv1D-LSTM + Contrastive + Recon + GRL + CondCORAL")
    print(f"  Window: {WINDOW_LEN}  Epochs: {EPOCHS}  LR: {LR}")
    print("=" * 80)
    print("Starting data loading...")

    # ----- Load HAR data (source = train, target = train_2, test = test) -----
    print("Loading source and test datasets...")
    src_ds, test_ds = load_csi_datasets([TRAIN_DIR],  [TEST_DIR], WINDOW_LEN, verbose=True)
    print(f"Source loaded: {src_ds.X.shape}")
    
    print("Loading target dataset...")
    try:
        tgt_ds, _ = load_csi_datasets([TRAIN_DIR2], [TEST_DIR], WINDOW_LEN, verbose=True)
        print(f"Target loaded: {tgt_ds.X.shape}")
    except Exception as e:
        print(f"Error loading target dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Re-encode target labels to match source label_map (maps may differ in index order)
    print("Re-encoding target labels...")
    reverse_tgt = {v: k for k, v in tgt_ds.label_map.items()}
    y_tgt = np.array([src_ds.label_map[reverse_tgt[yi]] for yi in tgt_ds.y])
    print("Target labels re-encoded")

    # ----- Load calibration data (unlabelled, for reconstruction) -----
    print("Loading calibration data...")
    calib_pipeline = Pipeline([
        CSI_Loader(verbose=False),
        FeatureSelector(verbose=False),
        WindowTransformer(window_length=WINDOW_LEN, key='mag', mode='flattened',
                          stride=WINDOW_LEN // 3, verbose=False),
    ])
    calib_csvs = sorted(glob.glob(f'{CALIBRATION_DIR}/*.csv'))
    print(f"Found {len(calib_csvs)} calibration files")
    calib_windows = []
    for csv_path in calib_csvs:
        data = calib_pipeline(csv_path)
        calib_windows.append(data['mag'])
    X_calib = np.concatenate(calib_windows, axis=0)
    print(f"Calibration data loaded: {X_calib.shape}")

    n_subcarriers = src_ds.X.shape[2]  # For sequential data
    n_classes = src_ds.num_classes
    idx2name = {v: k for k, v in src_ds.label_map.items()}

    # Flatten sequential data for model input
    if len(src_ds.X.shape) == 3:  # (samples, time, features)
        X_src_flat = src_ds.X.reshape(src_ds.X.shape[0], -1)
        X_test_flat = test_ds.X.reshape(test_ds.X.shape[0], -1)
        X_tgt_flat = tgt_ds.X.reshape(tgt_ds.X.shape[0], -1)
        print("Flattened sequential data for model input")
    else:
        X_src_flat = src_ds.X
        X_test_flat = test_ds.X  
        X_tgt_flat = tgt_ds.X
        n_subcarriers = src_ds.X.shape[1] // WINDOW_LEN

    print(f"  Source:  {src_ds.X.shape}  |  Target: {tgt_ds.X.shape}  |  Test: {test_ds.X.shape}")
    print(f"  Calib:   {X_calib.shape}  ({len(calib_csvs)} files from {CALIBRATION_DIR})")
    print(f"  Subcarriers: {n_subcarriers}  Classes: {n_classes}  Labels: {src_ds.label_map}")

    # ----- Build model -----
    print("Building model...")
    model = SeqModel(
        n_subcarriers=n_subcarriers,
        window_len=WINDOW_LEN,
        num_classes=n_classes,
        conv_channels=[64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
        use_whitening=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print("Model built successfully")

    # ----- Train -----
    print("Starting training...")
    model, info = train_seq_model(
        model,
        X_src=X_src_flat,   y_src=src_ds.y,
        X_tgt=X_tgt_flat,   y_tgt=y_tgt,
        X_test=X_test_flat,  y_test=test_ds.y,
        X_calib=X_calib,
        epochs=EPOCHS,
        batch_size=64,
        lr=LR,
        w_cls=1.0,
        w_recon=0.5,
        w_contrast=0.3,
        w_domain=0.3,
        w_coral=0.3,
        use_cond_coral=True,
        verbose=True,
    )

    # ----- Evaluate -----
    metrics = compute_metrics(model, X_test_flat, test_ds.y)
    print(f"\n{'='*60}")
    print("FINAL TEST METRICS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k == 'confusion_matrix':
            continue
        print(f"  {k:<22}: {v}")

    # Confusion matrix
    cm = metrics['confusion_matrix']
    cls_names = [idx2name.get(i, str(i)) for i in range(n_classes)]
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10}" + ''.join(f'{c:>8}' for c in cls_names))
    for i, row in enumerate(cm):
        print(f"  {cls_names[i]:>10}" + ''.join(f'{v:>8}' for v in row))

    print(f"\n{'='*80}")
    print("SEQ experiments completed!")
    print(f"{'='*80}")
