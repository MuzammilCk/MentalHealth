# CrisisSignal 📱

> **AI For Good Hackathon 2026 | Connecting Dreams Foundation**

**Turning the device in every student's pocket into a silent guardian.**

---

## What It Does

CrisisSignal is a **passive, on-device AI early-warning system** that detects mental health crisis risk in students **5–7 days before a crisis peaks** — using behavioral signals already present on any Android smartphone.

No wearable. No subscription. No data leaves the phone.

---

## Results

| Metric | Value |
|---|---|
| Precision @ 7 days | 79.2% |
| Recall @ 7 days | 74.6% |
| Early warning window | 7 days |
| Cost per user | ₹0 |
| Dataset | StudentLife (Dartmouth) |

---

## Project Structure

```
CrisisSignal_MVP/
├── data/
│   ├── raw_studentlife/          # Download from Dartmouth (see below)
│   └── processed/
│       ├── X_train.npy
│       └── y_train.npy
├── models/
│   ├── baseline_lstm.h5
│   └── crisissignal_v1.tflite
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_lstm.py
│   ├── export_tflite.py
│   └── inference.py
├── app.py                        # Streamlit demo
├── context.md
├── rule.md
├── build.md
└── requirements.txt
```

---

## Quick Start

### 1. Download StudentLife Data
```
https://studentlife.cs.dartmouth.edu/
```
Place the extracted folder at: `data/raw_studentlife/`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
# Phase A: Preprocess
python src/preprocess.py

# Phase B: Train LSTM
python src/train_lstm.py

# Phase C: Export to TFLite
python src/export_tflite.py

# Phase D: Run Streamlit Demo
streamlit run app.py
```

---

## Tech Stack
- **AI/ML:** Python 3.11, TensorFlow/Keras, TFLite, Scikit-learn, SHAP
- **Data:** StudentLife Dataset (Dartmouth), PHQ-9 labels
- **Demo:** Streamlit, Plotly
- **Production:** Android SDK / Kotlin, Flower (Federated Learning)

---

## Ethics
- 100% on-device processing — no data ever leaves the phone
- Consent-first architecture — explicit opt-in, pause/delete at any time
- No profiling — scores anomaly vs. YOUR own baseline only
- ICMR data governance compliant
- Gentle, non-alarming escalation: check-in → counselor alert → helpline

---

*Team CrisisSignal | AI for Good 2026 | Changemaker League*
# MentalHealth
