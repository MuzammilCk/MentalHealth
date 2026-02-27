"""
inference.py — Phase D: PyTorch Inference Engine for CrisisSignal
=================================================================
Loads the trained PyTorch LSTM Autoencoder from Phase B, runs reconstruction-
based anomaly detection on a 30-day window, and maps error to a 0-100 risk score.

This module is imported by app.py as the live inference backend.

Usage:
    from src.inference import CrisisSignalInference
    engine = CrisisSignalInference()
    risk_score = engine.predict_risk(window)   # ndarray (30, 5), values in [0,1]
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR     = PROJECT_ROOT / "models"
PROC_DIR       = PROJECT_ROOT / "data" / "processed"

MODEL_PATH     = MODELS_DIR / "baseline_lstm.pt"
ARCH_PATH      = MODELS_DIR / "baseline_lstm_arch.json"
TORCHSCRIPT_PATH = MODELS_DIR / "crisissignal_v1.ptl"   # preferred export (Phase C)
THRESHOLD_PATH = MODELS_DIR / "threshold.npy"
SCALER_PATH    = PROC_DIR   / "scaler.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 30
N_FEATURES  = 5
# Feature order (must match FEATURE_COLS in preprocess.py):
#   [sleep_duration_hours, conversation_frequency, location_variance,
#    activity_level, phone_usage_minutes]

_DEFAULT_MIN_ERR = 0.001
_DEFAULT_MAX_ERR = 0.15

# app.py passes values already normalized to [0,1] (slider_value / 100)
INPUT_ALREADY_NORMALIZED = True


# ── Internal model class (mirrors train_lstm.py) ──────────────────────────────
def _build_model(arch: dict):
    """Rebuild the LSTM Autoencoder from saved architecture metadata."""
    try:
        import torch
        import torch.nn as nn

        class LSTMAutoencoder(nn.Module):
            def __init__(self, n_features, hidden_size, latent_size):
                super().__init__()
                self.encoder_lstm = nn.LSTM(n_features, hidden_size,
                                            num_layers=1, batch_first=True)
                self.encoder_compress = nn.LSTM(hidden_size, latent_size,
                                                num_layers=1, batch_first=True)
                self.decoder_expand = nn.LSTM(latent_size, hidden_size,
                                              num_layers=1, batch_first=True)
                self.decoder_lstm = nn.LSTM(hidden_size, hidden_size,
                                            num_layers=1, batch_first=True)
                self.output_layer = nn.Linear(hidden_size, n_features)

            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                enc_out, _    = self.encoder_lstm(x)
                _, (h_n, _)   = self.encoder_compress(enc_out)
                latent        = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
                dec_out, _    = self.decoder_expand(latent)
                dec_out2, _   = self.decoder_lstm(dec_out)
                return self.output_layer(dec_out2)

        model = LSTMAutoencoder(
            n_features  = arch["n_features"],
            hidden_size = arch["hidden_size"],
            latent_size = arch["latent_size"],
        )
        return model, torch
    except ImportError:
        return None, None


# ── Inference Engine ──────────────────────────────────────────────────────────
class CrisisSignalInference:
    """
    Encapsulates the full CrisisSignal inference pipeline:
      1. Accept normalized [0,1] feature window (30 days × 5 features)
      2. Run PyTorch LSTM Autoencoder to reconstruct the sequence
      3. Compute mean squared reconstruction error (MSE)
      4. Map MSE to a 0–100 risk score using learned threshold calibration
    """

    def __init__(self) -> None:
        self._model     = None
        self._torch     = None
        self._scaler    = None
        self._threshold: float = 0.05
        self._min_err: float = _DEFAULT_MIN_ERR
        self._max_err: float = _DEFAULT_MAX_ERR
        self._model_loaded: bool = False

        self._load_model()
        self._load_scaler()
        self._load_threshold()

    # ── Loaders ───────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """
        Load inference model.
        Priority:
          1. TorchScript crisissignal_v1.ptl  (Phase C export — optimised, no class needed)
          2. Raw state dict baseline_lstm.pt  (Phase B output — requires class rebuild)
          3. Proxy fallback
        """
        # ── Option 1: TorchScript (preferred) ─────────────────────────────────
        if TORCHSCRIPT_PATH.exists():
            try:
                import torch
                scripted = torch.jit.load(str(TORCHSCRIPT_PATH), map_location="cpu")
                scripted.eval()
                self._model  = scripted
                self._torch  = torch
                self._model_loaded = True
                sz = TORCHSCRIPT_PATH.stat().st_size / 1024
                log.info(f"TorchScript model loaded from {TORCHSCRIPT_PATH}  ({sz:.1f} KB)")
                return
            except Exception as e:
                log.warning(f"TorchScript load failed ({e}), trying raw state dict...")

        # ── Option 2: Raw state dict ───────────────────────────────────────────
        if MODEL_PATH.exists() and ARCH_PATH.exists():
            try:
                with open(ARCH_PATH, "r") as f:
                    arch = json.load(f)
                model, torch = _build_model(arch)
                if model is None:
                    raise ImportError("PyTorch not available")
                state_dict = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self._model  = model
                self._torch  = torch
                self._model_loaded = True
                log.info(f"State-dict model loaded from {MODEL_PATH}")
                return
            except Exception as e:
                log.warning(f"State-dict load failed ({e}). Using proxy fallback.")

        log.warning(
            "No trained model found. Falling back to proxy risk calculation. "
            "Run Phase B + C to enable real inference."
        )

    def _load_scaler(self) -> None:
        """Load fitted MinMaxScaler from Phase A."""
        if not SCALER_PATH.exists():
            log.warning(f"Scaler not found at {SCALER_PATH}. Input not re-normalized.")
            return
        try:
            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            log.info(f"Scaler loaded from {SCALER_PATH}")
        except Exception as e:
            log.warning(f"Failed to load scaler: {e}")

    def _load_threshold(self) -> None:
        """Load optimal reconstruction error threshold from Phase B."""
        if not THRESHOLD_PATH.exists():
            log.warning(f"Threshold not found at {THRESHOLD_PATH}. Using default 0.05.")
            return
        try:
            data = np.load(str(THRESHOLD_PATH))
            self._threshold = float(data[0])
            self._min_err   = self._threshold * 0.1
            self._max_err   = self._threshold * 3.0
            log.info(f"Threshold loaded: {self._threshold:.6f}")
        except Exception as e:
            log.warning(f"Failed to load threshold: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_model_inference(self, window: np.ndarray) -> float:
        """
        Run PyTorch model on a single (30, 5) window.
        Returns MSE reconstruction error.
        """
        torch = self._torch
        x = torch.from_numpy(window[np.newaxis, ...].astype(np.float32))  # (1, 30, 5)
        with torch.no_grad():
            recon = self._model(x)
        mse = float(torch.mean((x - recon) ** 2).item())
        return mse

    def _error_to_risk_score(self, err: float) -> int:
        """Linear map: reconstruction error → 0–100 integer score."""
        if err <= self._min_err:
            return 0
        if err >= self._max_err:
            return 100
        normalized = (err - self._min_err) / (self._max_err - self._min_err)
        return int(round(normalized * 100))

    def predict_risk(self, feature_window: np.ndarray,
                     already_normalized: bool = INPUT_ALREADY_NORMALIZED) -> int:
        """
        Main inference entry point.

        Args:
            feature_window     : ndarray (30, 5), last 30 days of 5 features.
                                 Column order: [sleep_duration_hours,
                                 conversation_frequency, location_variance,
                                 activity_level, phone_usage_minutes]
            already_normalized : True (default) = input already in [0,1].
                                 app.py passes slider_values / 100, so True.

        Returns:
            risk_score : integer in [0, 100]
        """
        if feature_window.shape != (WINDOW_SIZE, N_FEATURES):
            raise ValueError(
                f"Expected shape ({WINDOW_SIZE}, {N_FEATURES}), "
                f"got {feature_window.shape}"
            )

        # Clip to [0,1] if already normalized, else apply scaler
        if already_normalized:
            window_scaled = np.clip(feature_window, 0.0, 1.0).astype(np.float32)
        elif self._scaler is not None:
            flat = feature_window.reshape(-1, N_FEATURES)
            scaled = self._scaler.transform(flat)
            window_scaled = np.clip(scaled, 0.0, 1.0).reshape(WINDOW_SIZE, N_FEATURES).astype(np.float32)
        else:
            window_scaled = feature_window.astype(np.float32)

        if self._model_loaded:
            err = self._run_model_inference(window_scaled)
            risk = self._error_to_risk_score(err)
            log.debug(f"MSE: {err:.6f} → Risk: {risk}")
            return risk
        else:
            return self._proxy_risk_score(window_scaled)

    def _proxy_risk_score(self, window_scaled: np.ndarray) -> int:
        """
        Fallback proxy when model unavailable.
        Uses last day's values and published feature weights from pitch deck.
        Expects window in [0,1].
        """
        weights = np.array([0.91, 0.78, 0.71, 0.65, 0.58])
        last_day = window_scaled[-1]   # most recent day
        raw = float(np.dot(last_day, weights) / weights.sum())
        return int(np.clip(raw * 100, 0, 100))

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def is_using_real_model(self) -> bool:
        """True if PyTorch model is loaded and active."""
        return self._model_loaded

    @property
    def threshold(self) -> float:
        """Reconstruction error threshold (healthy/at-risk boundary)."""
        return self._threshold
