"""
train_lstm.py — Phase B: LSTM Autoencoder Training for CrisisSignal
====================================================================
Uses PyTorch (Python 3.13 compatible) to train an LSTM Autoencoder.
TensorFlow is NOT required for training — only PyTorch is needed.

Strategy (unchanged from design doc):
  - Train ONLY on healthy students (label=0) → model learns normal behavior
  - At inference: high reconstruction error = behavioral anomaly (at-risk)
  - Find optimal MSE threshold that maximizes F1 vs. depressed samples

Output:
  models/baseline_lstm.pt          — PyTorch LSTM Autoencoder state dict
  models/baseline_lstm_arch.json   — Architecture hyperparams (for inference)
  models/threshold.npy             — Optimal reconstruction error threshold
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PROC_DIR       = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"

X_PATH         = PROC_DIR  / "X_train.npy"
Y_PATH         = PROC_DIR  / "y_train.npy"
MODEL_PATH     = MODELS_DIR / "baseline_lstm.pt"
ARCH_PATH      = MODELS_DIR / "baseline_lstm_arch.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.npy"

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_SIZE   = 64
LATENT_SIZE   = 32
DROPOUT       = 0.2
BATCH_SIZE    = 32
MAX_EPOCHS    = 100
PATIENCE      = 7       # epochs without improvement before early stop
LR            = 1e-3
VAL_SPLIT     = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.

    Encoder  : LSTM(hidden) → LSTM(latent) → bottleneck
    Decoder  : repeat → LSTM(latent) → LSTM(hidden) → Linear(n_features)
    """

    def __init__(self, n_features: int, hidden_size: int, latent_size: int,
                 n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_layers    = n_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder_compress = nn.LSTM(
            input_size=hidden_size,
            hidden_size=latent_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Decoder
        self.decoder_expand = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            reconstruction: (batch, seq_len, n_features)
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        enc_out, _ = self.encoder_lstm(x)           # (B, T, H)
        enc_out    = self.dropout(enc_out)
        enc_out2, (h_n, _) = self.encoder_compress(enc_out)  # (B, T, L)

        # Bottleneck: use last hidden state, repeat for decoder
        latent = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, L)

        # Decode
        dec_out, _ = self.decoder_expand(latent)    # (B, T, H)
        dec_out    = self.dropout(dec_out)
        dec_out2, _ = self.decoder_lstm(dec_out)    # (B, T, H)
        reconstruction = self.output_layer(dec_out2)  # (B, T, n_features)

        return reconstruction


# ── Training Helpers ──────────────────────────────────────────────────────────
def compute_mse_per_sample(model: nn.Module, X: np.ndarray,
                            batch_size: int = 64) -> np.ndarray:
    """Compute per-sample MSE reconstruction error (no grad)."""
    model.eval()
    errors = []
    n = len(X)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).to(DEVICE)
            recon = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors).astype(np.float32)


def find_optimal_threshold(errors_healthy: np.ndarray,
                           errors_depressed: np.ndarray) -> float:
    """Sweep candidate thresholds and return the one that maximizes F1."""
    all_errors = np.concatenate([errors_healthy, errors_depressed])
    all_labels = np.array(
        [0] * len(errors_healthy) + [1] * len(errors_depressed), dtype=int
    )

    best_f1, best_thresh = 0.0, float(np.percentile(errors_healthy, 95))

    for thresh in np.percentile(all_errors, np.arange(5, 96, 1)):
        preds = (all_errors >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)

    log.info(f"Optimal threshold: {best_thresh:.6f}  (F1 = {best_f1:.4f})")
    return best_thresh


# ── Main Training Pipeline ────────────────────────────────────────────────────
def run_training() -> None:
    """Full Phase B training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Using device: {DEVICE}")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not X_PATH.exists() or not Y_PATH.exists():
        log.error(
            "Preprocessed data not found.\n"
            "Run Phase A first:  python src/preprocess.py"
        )
        raise FileNotFoundError(f"Missing: {X_PATH} or {Y_PATH}")

    X = np.load(str(X_PATH)).astype(np.float32)
    y = np.load(str(Y_PATH)).astype(np.int32)
    log.info(f"Loaded X: {X.shape}, y: {y.shape}")

    n_samples, timesteps, n_features = X.shape

    # ── Split healthy vs depressed ─────────────────────────────────────────────
    X_healthy   = X[y == 0]
    X_depressed = X[y == 1]

    log.info(f"Healthy samples    : {len(X_healthy)}")
    log.info(f"Depressed samples  : {len(X_depressed)}")

    if len(X_healthy) < BATCH_SIZE:
        raise ValueError(
            f"Only {len(X_healthy)} healthy samples — too few for training. "
            "Check Phase A output."
        )

    # Train only on healthy (normal) behavior
    X_train, X_val = train_test_split(
        X_healthy, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    log.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    criterion = nn.MSELoss()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(X: np.ndarray, shuffle: bool) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=0, pin_memory=(DEVICE.type == "cuda")
        )

    train_loader = make_loader(X_train, shuffle=True)
    val_loader   = make_loader(X_val,   shuffle=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("Starting training...")
    best_val_loss   = float("inf")
    best_state_dict = None
    patience_count  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                loss  = criterion(recon, batch)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    log.info(f"Best val loss: {best_val_loss:.6f}")

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ── Evaluate & find threshold ─────────────────────────────────────────────
    log.info("Computing reconstruction errors for evaluation...")
    errors_healthy   = compute_mse_per_sample(model, X_healthy)
    errors_depressed = compute_mse_per_sample(model, X_depressed) if len(X_depressed) > 0 else np.array([])

    if len(errors_depressed) > 0:
        threshold = find_optimal_threshold(errors_healthy, errors_depressed)

        all_errors = np.concatenate([errors_healthy, errors_depressed])
        all_labels = np.array(
            [0] * len(errors_healthy) + [1] * len(errors_depressed), dtype=int
        )
        predictions = (all_errors >= threshold).astype(int)

        precision = precision_score(all_labels, predictions, zero_division=0)
        recall    = recall_score(all_labels, predictions, zero_division=0)

        report_str = classification_report(
            all_labels, predictions,
            target_names=["Healthy", "Depressed"]
        )

        log.info("\n" + "=" * 55)
        log.info("CLASSIFICATION REPORT")
        log.info("=" * 55)
        log.info(f"\n{report_str}")
        log.info(f"Precision @ Threshold : {precision * 100:.1f}%")
        log.info(f"Recall    @ Threshold : {recall * 100:.1f}%")
        log.info("=" * 55 + "\n")
    else:
        log.warning("No depressed samples — using 95th percentile as threshold.")
        threshold = float(np.percentile(errors_healthy, 95))

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Save model weights
    torch.save(model.state_dict(), str(MODEL_PATH))

    # Save architecture metadata (needed by inference.py to rebuild model)
    arch = {
        "n_features":  n_features,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "dropout":     0.0,   # inference always uses dropout=0
        "timesteps":   timesteps,
    }
    with open(ARCH_PATH, "w") as f:
        json.dump(arch, f, indent=2)

    # Save threshold
    np.save(str(THRESHOLD_PATH), np.array([threshold], dtype=np.float32))

    log.info(f"Saved model      → {MODEL_PATH}")
    log.info(f"Saved arch JSON  → {ARCH_PATH}")
    log.info(f"Saved threshold  → {THRESHOLD_PATH}  (value: {threshold:.6f})")
    log.info("✅ Phase B COMPLETE")


if __name__ == "__main__":
    run_training()
