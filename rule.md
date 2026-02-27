# CrisisSignal — AI Coding Rules & Constraints

## Core Constraints (MUST follow at all times)

### ❌ NO Hallucination
- Do NOT invent dataset column names, file paths, or API signatures that you have not verified
- If a StudentLife CSV column name is uncertain, use a safe fallback with a clear comment explaining where to find the correct name
- Do NOT fabricate model metrics — all numbers must come from actual training output

### ❌ NO Hardcoded Values
- Do NOT hardcode file paths — use `os.path.join` and `pathlib.Path` with relative paths from project root
- Do NOT hardcode threshold values without defining them as named constants at the top of the file
- Do NOT hardcode model weights or feature importances that should be computed from data

### ❌ NO Fake/Placeholder Code
- Do NOT write `# TODO: implement this` stubs and present them as done
- Do NOT use `np.random` to simulate "real" data without clearly labeling it as simulation
- Every function must be fully implemented and runnable

### ✅ MUST Stay in Context
- All code must serve the CrisisSignal pipeline described in context.md
- Do not add features not in the phases or requested by the user
- Do not use libraries not listed in the tech stack unless strictly necessary, and if so, add them to requirements.txt

### ✅ MUST Be Reproducible
- Every script must be runnable end-to-end with a single `python <script>.py` command
- Seed all random operations: `np.random.seed(42)`, `tf.random.set_seed(42)`
- All intermediate artifacts (`.npy`, `.h5`, `.tflite`) must be saved to deterministic paths

### ✅ MUST Have Error Handling
- All file I/O must use try-except blocks with descriptive error messages
- If the StudentLife data directory is missing, print a clear message with the download URL
- Model training must gracefully handle shape mismatches

### ✅ Code Quality Standards
- Use type hints on all function signatures
- Use docstrings on all functions
- Follow PEP-8 formatting
- Use logging module (not print) for training progress in production scripts
- All scripts must have a `if __name__ == '__main__':` guard

## File & Path Conventions
```
Project root: d:\projects\Mental Health\
Data:         data/raw_studentlife/     (user downloads manually)
Processed:    data/processed_sequences.npy
Models:       models/baseline_lstm.h5
              models/crisissignal_v1.tflite
Source:       src/preprocess.py
              src/train_lstm.py
              src/export_tflite.py
              src/inference.py
Notebooks:    notebooks/01_EDA.ipynb
App:          app.py
```

## Phase Completion Gate
Before marking any phase done:
1. The script runs without errors on the developer machine
2. The expected output file(s) are created
3. build.md is updated with actual output and any deviations from plan
