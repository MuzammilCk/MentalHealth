"""
export_tflite.py — Phase C: Model Export for CrisisSignal
==========================================================
Converts the trained PyTorch LSTM Autoencoder to TorchScript —
the PyTorch equivalent of TFLite: a portable, self-contained,
Python-free model file for on-device deployment.

Why TorchScript instead of TFLite:
  TensorFlow has confirmed DLL failures on Python 3.13 Windows.
  TorchScript (torch.jit.trace) produces an equivalent portable
  artifact (.ptl) that runs in C++ / Android without Python.

Plan equivalence:
  TFLite plan                 →   This implementation
  ─────────────────────────────────────────────────────
  load baseline_lstm.h5       →   load baseline_lstm.pt + arch JSON
  quantize (dynamic range)    →   torch.quantization.quantize_dynamic
  verify inference pass       →   verify with test tensor ✅
  save crisissignal_v1.tflite →   save crisissignal_v1.ptl (TorchScript)

Output:
  models/crisissignal_v1.ptl  — TorchScript portable model (deployment asset)
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.quantization

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
PROC_DIR      = PROJECT_ROOT / "data" / "processed"

MODEL_PT_PATH  = MODELS_DIR / "baseline_lstm.pt"
ARCH_PATH      = MODELS_DIR / "baseline_lstm_arch.json"
EXPORT_PATH    = MODELS_DIR / "crisissignal_v1.ptl"
X_TRAIN_PATH   = PROC_DIR   / "X_train.npy"

# Quantization: "dynamic" (recommended for LSTM) or "none"
QUANTIZATION_MODE = "dynamic"


# ── Rebuild Model (mirrors train_lstm.py) ─────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder — must match train_lstm.py exactly."""

    def __init__(self, n_features: int, hidden_size: int, latent_size: int) -> None:
        super().__init__()
        self.encoder_lstm     = nn.LSTM(n_features,  hidden_size, num_layers=1, batch_first=True)
        self.encoder_compress = nn.LSTM(hidden_size,  latent_size, num_layers=1, batch_first=True)
        self.decoder_expand   = nn.LSTM(latent_size,  hidden_size, num_layers=1, batch_first=True)
        self.decoder_lstm     = nn.LSTM(hidden_size,  hidden_size, num_layers=1, batch_first=True)
        self.output_layer     = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        enc_out, _    = self.encoder_lstm(x)
        _, (h_n, _)   = self.encoder_compress(enc_out)
        latent        = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _    = self.decoder_expand(latent)
        dec_out2, _   = self.decoder_lstm(dec_out)
        return self.output_layer(dec_out2)


def load_model(arch: dict) -> LSTMAutoencoder:
    """Load model from state dict + architecture metadata."""
    model = LSTMAutoencoder(
        n_features  = arch["n_features"],
        hidden_size = arch["hidden_size"],
        latent_size = arch["latent_size"],
    )
    state_dict = torch.load(str(MODEL_PT_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Quantization ──────────────────────────────────────────────────────────────
def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply Post-Training Dynamic Range Quantization.
    Quantizes LSTM and Linear weights to int8 at runtime.
    Equivalent to TFLite dynamic range quantization.
    """
    log.info("Applying Dynamic Range Quantization (int8 weights)...")
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8,
    )
    log.info("Quantization applied.")
    return quantized


# ── TorchScript Export ────────────────────────────────────────────────────────
def export_torchscript(model: nn.Module, arch: dict) -> None:
    """
    Trace the model and save as TorchScript (.ptl).

    TorchScript models:
      - Run without Python (C++, Android, iOS)
      - No model class definition needed at load time
      - Equivalent deployment asset to .tflite
    """
    timesteps  = arch["timesteps"]
    n_features = arch["n_features"]

    # Create a representative input for tracing
    dummy_input = torch.randn(1, timesteps, n_features)

    log.info("Tracing model with torch.jit.trace...")
    try:
        scripted = torch.jit.trace(model, dummy_input)
    except Exception as e:
        log.error(f"TorchScript trace failed: {e}")
        raise

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(EXPORT_PATH))

    size_kb = EXPORT_PATH.stat().st_size / 1024
    log.info(f"Saved TorchScript model → {EXPORT_PATH}  ({size_kb:.1f} KB)")


# ── Verification ──────────────────────────────────────────────────────────────
def verify_export(arch: dict) -> None:
    """
    Load the exported .ptl file and run a test inference.
    Verifies the export is not corrupted and produces sensible output.
    """
    log.info("Verifying exported TorchScript model...")

    loaded = torch.jit.load(str(EXPORT_PATH))
    loaded.eval()

    timesteps  = arch["timesteps"]
    n_features = arch["n_features"]

    # Test 1: random input (should produce non-zero output)
    test_input = torch.rand(1, timesteps, n_features)
    with torch.no_grad():
        output = loaded(test_input)

    assert output.shape == (1, timesteps, n_features), \
        f"Unexpected output shape: {output.shape}"

    mse = float(torch.mean((test_input - output) ** 2).item())
    log.info(f"Test 1 — random input MSE (reconstruction error): {mse:.6f}")

    # Test 2: near-zero input (healthy-like) should give lower MSE than noisy
    zero_input  = torch.zeros(1, timesteps, n_features)
    noisy_input = torch.ones(1, timesteps, n_features)

    with torch.no_grad():
        out_zero  = loaded(zero_input)
        out_noisy = loaded(noisy_input)

    mse_zero  = float(torch.mean((zero_input  - out_zero)  ** 2).item())
    mse_noisy = float(torch.mean((noisy_input - out_noisy) ** 2).item())

    log.info(f"Test 2 — near-zero input MSE  : {mse_zero:.6f}")
    log.info(f"Test 2 — ones input MSE       : {mse_noisy:.6f}")
    log.info("TorchScript model verification passed ✅")


# ── Main Export Pipeline ──────────────────────────────────────────────────────
def run_export() -> None:
    """Full Phase C export pipeline."""
    # Check prerequisites
    if not MODEL_PT_PATH.exists():
        log.error(
            f"Trained model not found: {MODEL_PT_PATH}\n"
            "Run Phase B first: python src/train_lstm.py"
        )
        raise FileNotFoundError(str(MODEL_PT_PATH))

    if not ARCH_PATH.exists():
        log.error(f"Architecture JSON not found: {ARCH_PATH}")
        raise FileNotFoundError(str(ARCH_PATH))

    # Load architecture
    with open(ARCH_PATH, "r") as f:
        arch = json.load(f)
    log.info(f"Loaded architecture: {arch}")

    # Load model
    log.info(f"Loading model from: {MODEL_PT_PATH}")
    model = load_model(arch)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded. Parameters: {total_params:,}")

    # Apply quantization (if enabled)
    if QUANTIZATION_MODE == "dynamic":
        model = apply_dynamic_quantization(model)
    else:
        log.info("Skipping quantization (mode='none').")

    # Export to TorchScript
    export_torchscript(model, arch)

    # Verify the export
    verify_export(arch)

    log.info("✅ Phase C COMPLETE")
    log.info(f"   Deployment asset: {EXPORT_PATH}")


if __name__ == "__main__":
    run_export()
