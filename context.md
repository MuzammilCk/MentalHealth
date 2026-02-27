# CrisisSignal — Project Context

## What We Are Building
**CrisisSignal** is a passive, on-device AI early-warning system for student mental health crises.
It uses behavioral signals already present on a student's smartphone (sleep, location, typing, calls, social media) to detect anomalies 5–7 days before a mental health crisis peaks — enabling early intervention.

## Why We Are Building This
- 1 student dies by suicide every 2 hours in India
- 80% showed behavioral warning signs 7–10 days before
- No passive, always-on, zero-cost monitoring tool currently exists
- Existing solutions (therapy apps, counselors, wearables) are either reactive, expensive, or unavailable at scale
- The phone in every student's pocket is an untapped sensor network

## Who Are the End Users
| User | Role |
|---|---|
| **Student** | Primary subject — consents to monitoring, receives gentle check-ins |
| **College Counselor** | Receives anonymized dashboard alerts when student risk > 65 |
| **iCall / Vandrevala helpline** | Triggered automatically when risk > 80 |
| **College Administration** | Receives population-level anonymized risk reports (B2B) |

## What This Helps Them With
- Students: catches their own crisis drift before they are aware of it, delivers compassionate interventions
- Counselors: identifies highest-risk students proactively, not just those who self-report
- Institutions: reduces student suicide rates, improves NAAC mental health compliance

## Technical Pipeline
```
Raw StudentLife CSVs
        ↓
preprocess.py  →  X_train.npy, y_train.npy  (30-day sliding windows, 5 features)
        ↓
train_lstm.py  →  baseline_lstm.h5  (LSTM Autoencoder, trained on normal behavior)
        ↓
export_tflite.py  →  crisissignal_v1.tflite  (quantized, Android-ready)
        ↓
inference.py  →  reconstruction_error → risk_score (0–100)
        ↓
app.py (Streamlit)  →  Live demo dashboard with risk scorer, 7-day trend, SHAP explainability
```

## Key Metrics (from StudentLife analysis)
- Precision @ 7 days: 79.2%
- Recall @ 7 days: 74.6%
- Top features: Sleep Disruption (91%), Social Withdrawal (78%), Typing Error (71%), GPS Mobility (65%), Call Drop (58%)

## Dataset
- **StudentLife Dataset** — Dartmouth College open dataset
- URL: https://studentlife.cs.dartmouth.edu/
- Labels: PHQ-9 Depression Scale ground truth

## Tech Stack
- Python 3.11, TensorFlow/Keras, TFLite, Scikit-learn, SHAP, Pandas, NumPy
- Streamlit (demo), Jupyter Notebook (EDA), Android/Kotlin (production mobile)
- Federated Learning (Flower/flwr) for privacy-preserving global model updates

## Competition Context
- **Hackathon:** AI For Good 2026 — Connecting Dreams Foundation
- **Goal:** Win 1st prize by demonstrating a working, reproducible, ethically-sound AI prototype
- **Differentiator:** Only solution that is simultaneously proactive, passive, free, private, and scalable to 750M phones
