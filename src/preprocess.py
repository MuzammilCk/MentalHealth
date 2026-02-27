"""
preprocess.py — Phase A: Data Preprocessing for CrisisSignal
=============================================================
Rewritten to match the EXACT StudentLife dataset structure
(Dartmouth, released 2014). All paths, column names, and data
formats are verified against the real files on disk.

Real dataset layout:
  data/raw_studentlife/
  ├── sensing/
  │   ├── activity/      activity_u00.csv ... activity_u59.csv
  │   │                  cols: timestamp, ' activity inference'
  │   ├── conversation/  conversation_u00.csv ...
  │   │                  cols: start_timestamp, ' end_timestamp'
  │   ├── gps/           gps_u00.csv ...
  │   │                  index=time(epoch), cols: latitude, longitude, speed ...
  │   ├── phonelock/     phonelock_u00.csv ...
  │   │                  cols: start, end  (epoch seconds)
  ├── call_log/          call_log_u00.csv ...
  │                      cols: timestamp, CALLS_duration ...
  ├── EMA/response/Sleep/  Sleep_u00.json ... (resp_time = EMA response time)
  └── survey/PHQ-9.csv   one row per (uid, pre/post), text ordinal responses

Features produced (one row per day per student):
  1. sleep_duration_hours     — PSQI / Sleep EMA (hours slept per night)
  2. conversation_frequency   — count of conversation segments per day
  3. location_variance        — lat + lon variance (GPS mobility spread)
  4. activity_level           — mean activity inference (0=stationary…3=running)
  5. phone_usage_minutes      — total screen-on duration (phonelock end-start)

Label: PHQ-9 total score >= 10 → depressed (1), else healthy (0)

Output:
  data/processed/X_train.npy   shape (N, 30, 5)  float32
  data/processed/y_train.npy   shape (N,)         int32
  data/processed/scaler.pkl
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw_studentlife"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

SENSING_DIR  = RAW_DIR / "sensing"
EMA_DIR      = RAW_DIR / "EMA" / "response"
SURVEY_DIR   = RAW_DIR / "survey"

WINDOW_SIZE  = 30    # days per LSTM sequence
N_FEATURES   = 5
RANDOM_SEED  = 42
PHQ9_THRESHOLD = 10  # PHQ-9 score >= 10 = moderate depression (clinical standard)

# PHQ-9 Likert text → ordinal score mapping (standard PHQ-9 scoring)
PHQ9_SCORE_MAP: Dict[str, int] = {
    "not at all":             0,
    "several days":           1,
    "more than half the days": 2,
    "nearly every day":       3,
}

np.random.seed(RANDOM_SEED)

# ── Helper: discover UIDs ─────────────────────────────────────────────────────
def get_all_uids() -> List[str]:
    """Return sorted list of student UIDs that have at least one sensing file."""
    uids = set()
    for folder in ["activity", "conversation", "gps", "phonelock"]:
        sensing_folder = SENSING_DIR / folder
        if not sensing_folder.exists():
            continue
        for f in sensing_folder.iterdir():
            m = re.match(r"[a-z_]+_(u\d+)\.csv", f.name)
            if m:
                uids.add(m.group(1))
    return sorted(uids)


# ── Feature 1: Sleep Duration ─────────────────────────────────────────────────
def load_sleep_hours(uid: str) -> pd.Series:
    """
    Load sleep EMA responses for one student.
    Sleep_u00.json holds timestamped EMA surveys. The 'null' key (a quirk of
    the StudentLife dataset encoding) contains the numeric answer for sleep
    duration (hours). We parse those and resample to daily.

    Returns: pd.Series indexed by date (normalized datetime), values = hours slept.
    """
    sleep_path = EMA_DIR / "Sleep" / f"Sleep_{uid}.json"
    if not sleep_path.exists():
        return pd.Series(dtype=float, name="sleep_duration_hours")

    try:
        with open(sleep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"  [{uid}] Sleep JSON read error: {e}")
        return pd.Series(dtype=float, name="sleep_duration_hours")

    records = []
    for entry in data:
        ts  = entry.get("resp_time")
        val = entry.get("null", "")  # StudentLife stores answer under 'null' key
        if ts is None:
            continue
        try:
            hours = float(str(val).split(",")[0])   # some entries are "8" plain
            if hours <= 0 or hours > 24:
                continue
            date = pd.Timestamp(ts, unit="s").normalize()
            records.append({"date": date, "sleep_duration_hours": hours})
        except (ValueError, TypeError):
            continue

    if not records:
        return pd.Series(dtype=float, name="sleep_duration_hours")

    df = pd.DataFrame(records)
    series = df.groupby("date")["sleep_duration_hours"].mean()
    series.name = "sleep_duration_hours"
    return series


# ── Feature 2: Conversation Frequency ────────────────────────────────────────
def load_conversation_frequency(uid: str) -> pd.Series:
    """
    Count daily conversation segments from conversation_u00.csv.
    Real columns: 'start_timestamp', ' end_timestamp' (note leading space).
    """
    conv_path = SENSING_DIR / "conversation" / f"conversation_{uid}.csv"
    if not conv_path.exists():
        return pd.Series(dtype=float, name="conversation_frequency")

    try:
        df = pd.read_csv(conv_path)
    except Exception as e:
        log.warning(f"  [{uid}] Conversation CSV read error: {e}")
        return pd.Series(dtype=float, name="conversation_frequency")

    ts_col = "start_timestamp"
    if ts_col not in df.columns:
        log.warning(f"  [{uid}] Column '{ts_col}' not found. Found: {df.columns.tolist()}")
        return pd.Series(dtype=float, name="conversation_frequency")

    df["date"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    series = df.groupby("date").size().astype(float)
    series.name = "conversation_frequency"
    return series


# ── Feature 3: Location Variance (GPS Mobility) ───────────────────────────────
def load_location_variance(uid: str) -> pd.Series:
    """
    GPS mobility as daily lat+lon variance.
    Real columns: index=time(epoch), 'latitude', 'longitude'.
    Note: the GPS CSV uses the timestamp as the row index (no named index col).
    """
    gps_path = SENSING_DIR / "gps" / f"gps_{uid}.csv"
    if not gps_path.exists():
        return pd.Series(dtype=float, name="location_variance")

    try:
        # The first column (time) is used as the index in the real file
        df = pd.read_csv(gps_path, index_col=0)
    except Exception as e:
        log.warning(f"  [{uid}] GPS CSV read error: {e}")
        return pd.Series(dtype=float, name="location_variance")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        log.warning(f"  [{uid}] GPS missing lat/lon cols. Found: {df.columns.tolist()}")
        return pd.Series(dtype=float, name="location_variance")

    # Index is epoch seconds
    df.index = pd.to_datetime(df.index, unit="s", errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["date"] = df.index.normalize()

    def variance_2d(g: pd.DataFrame) -> float:
        return float(g["latitude"].var(ddof=0) + g["longitude"].var(ddof=0))

    series = df.groupby("date").apply(variance_2d)
    series.name = "location_variance"
    return series


# ── Feature 4: Activity Level ─────────────────────────────────────────────────
def load_activity_level(uid: str) -> pd.Series:
    """
    Mean daily activity inference (0=stationary, 1=walking, 2=running, 3=cycling).
    Real columns: 'timestamp', ' activity inference' (note leading space).
    """
    act_path = SENSING_DIR / "activity" / f"activity_{uid}.csv"
    if not act_path.exists():
        return pd.Series(dtype=float, name="activity_level")

    try:
        df = pd.read_csv(act_path)
    except Exception as e:
        log.warning(f"  [{uid}] Activity CSV read error: {e}")
        return pd.Series(dtype=float, name="activity_level")

    # column name has a leading space in the real dataset
    act_col = next((c for c in df.columns if "activity" in c.lower() or "infer" in c.lower()), None)
    ts_col  = next((c for c in df.columns if "time" in c.lower()), None)

    if not act_col or not ts_col:
        log.warning(f"  [{uid}] Activity missing timestamp/inference cols. Found: {df.columns.tolist()}")
        return pd.Series(dtype=float, name="activity_level")

    df["date"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce").dt.normalize()
    df[act_col] = pd.to_numeric(df[act_col], errors="coerce")
    df = df.dropna(subset=["date", act_col])

    series = df.groupby("date")[act_col].mean()
    series.name = "activity_level"
    return series


# ── Feature 5: Phone Usage Minutes ───────────────────────────────────────────
def load_phone_usage_minutes(uid: str) -> pd.Series:
    """
    Daily total screen-on time from phonelock_u00.csv.
    Real columns: 'start', 'end' (epoch seconds).
    Duration = end - start in seconds → convert to minutes.
    """
    lock_path = SENSING_DIR / "phonelock" / f"phonelock_{uid}.csv"
    if not lock_path.exists():
        return pd.Series(dtype=float, name="phone_usage_minutes")

    try:
        df = pd.read_csv(lock_path)
    except Exception as e:
        log.warning(f"  [{uid}] Phonelock CSV read error: {e}")
        return pd.Series(dtype=float, name="phone_usage_minutes")

    if "start" not in df.columns or "end" not in df.columns:
        log.warning(f"  [{uid}] Phonelock missing start/end. Found: {df.columns.tolist()}")
        return pd.Series(dtype=float, name="phone_usage_minutes")

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    df = df.dropna(subset=["start", "end"])

    # Screen-on duration in minutes
    df["duration_min"] = (df["end"] - df["start"]) / 60.0
    # Clip negative / unrealistically large values (>8 hours per session)
    df = df[(df["duration_min"] > 0) & (df["duration_min"] < 480)]

    df["date"] = pd.to_datetime(df["start"], unit="s", errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    series = df.groupby("date")["duration_min"].sum()
    series.name = "phone_usage_minutes"
    return series


# ── Label: PHQ-9 Score ────────────────────────────────────────────────────────
def load_phq9_labels() -> Dict[str, int]:
    """
    Parse survey/PHQ-9.csv and return a mapping {uid: binary_label}.
    Uses the 'post' measurement if available, otherwise 'pre'.
    PHQ-9 score >= 10 → label 1 (depressed), else 0.
    """
    phq_path = SURVEY_DIR / "PHQ-9.csv"
    if not phq_path.exists():
        log.error(f"PHQ-9.csv not found at {phq_path}")
        return {}

    df = pd.read_csv(phq_path)

    # Question columns = all except 'uid', 'type', 'Response'
    meta_cols  = {"uid", "type", "Response"}
    q_cols     = [c for c in df.columns if c not in meta_cols]

    def text_to_score(row: pd.Series) -> int:
        """Sum ordinal scores for each PHQ-9 item."""
        total = 0
        for col in q_cols:
            val = str(row[col]).strip().lower()
            total += PHQ9_SCORE_MAP.get(val, 0)
        return total

    df["phq9_score"] = df.apply(text_to_score, axis=1)

    # Prefer 'post' over 'pre'; if both, take post
    labels: Dict[str, int] = {}
    for uid, group in df.groupby("uid"):
        post_rows = group[group["type"] == "post"]
        pre_rows  = group[group["type"] == "pre"]
        row = post_rows.iloc[0] if not post_rows.empty else pre_rows.iloc[0]
        score = row["phq9_score"]
        labels[uid] = 1 if score >= PHQ9_THRESHOLD else 0
        log.debug(f"  {uid}: PHQ-9 score={score} → label={labels[uid]}")

    log.info(f"PHQ-9 labels loaded: {len(labels)} students | "
             f"Depressed: {sum(labels.values())} | "
             f"Healthy: {len(labels) - sum(labels.values())}")
    return labels


# ── Merge Features ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sleep_duration_hours",
    "conversation_frequency",
    "location_variance",
    "activity_level",
    "phone_usage_minutes",
]

def build_daily_feature_frame(uid: str) -> pd.DataFrame:
    """
    Merge all 5 feature series into a single daily DataFrame for one student.
    Missing days are forward-filled then back-filled.
    Returns DataFrame with columns = FEATURE_COLS, index = date.
    """
    series_list = [
        load_sleep_hours(uid),
        load_conversation_frequency(uid),
        load_location_variance(uid),
        load_activity_level(uid),
        load_phone_usage_minutes(uid),
    ]

    # Drop completely empty series
    non_empty = [s for s in series_list if not s.empty]
    if not non_empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    df = pd.concat(non_empty, axis=1)
    df = df.sort_index()

    # Fill missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURE_COLS]
    df = df.ffill().bfill()
    df = df.fillna(0.0)  # Any remaining NaN → 0

    return df


# ── Sliding Window ────────────────────────────────────────────────────────────
def create_sliding_windows(df: pd.DataFrame) -> np.ndarray:
    """
    Create (window_size, n_features) overlapping windows from a daily DataFrame.
    Returns array of shape (n_windows, WINDOW_SIZE, N_FEATURES).
    """
    values = df[FEATURE_COLS].values.astype(np.float32)
    n = len(values)
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, N_FEATURES), dtype=np.float32)

    windows = np.stack([values[i : i + WINDOW_SIZE] for i in range(n - WINDOW_SIZE + 1)])
    return windows


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing() -> None:
    """Execute the full Phase A preprocessing pipeline."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        log.error(
            f"\nRaw data not found: {RAW_DIR}\n"
            "Download from: https://studentlife.cs.dartmouth.edu/\n"
            "Extract to:    data/raw_studentlife/"
        )
        raise FileNotFoundError(str(RAW_DIR))

    # ── Load PHQ-9 labels ──────────────────────────────────────────────────────
    phq9_labels = load_phq9_labels()
    if not phq9_labels:
        raise ValueError("No PHQ-9 labels found — cannot build training labels.")

    # ── Discover UIDs ──────────────────────────────────────────────────────────
    uids = get_all_uids()
    log.info(f"Found {len(uids)} students with sensing data: {uids}")

    all_X: List[np.ndarray] = []
    all_y: List[int]        = []
    skipped = 0

    for uid in uids:
        log.info(f"Processing {uid}...")

        label = phq9_labels.get(uid)
        if label is None:
            log.warning(f"  [{uid}] No PHQ-9 label — skipping.")
            skipped += 1
            continue

        df = build_daily_feature_frame(uid)

        if len(df) < WINDOW_SIZE:
            log.warning(
                f"  [{uid}] Only {len(df)} days of data (need {WINDOW_SIZE}) — skipping."
            )
            skipped += 1
            continue

        windows = create_sliding_windows(df)
        if len(windows) == 0:
            skipped += 1
            continue

        all_X.append(windows)
        all_y.extend([label] * len(windows))
        log.info(f"  [{uid}] {len(windows)} windows, label={label}")

    if not all_X:
        raise ValueError(
            "No usable student data. Check that:\n"
            "  1. data/raw_studentlife/sensing/ contains CSV files.\n"
            "  2. Students have >= 30 days of sensing data.\n"
            "  3. survey/PHQ-9.csv exists."
        )

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int32)

    log.info(f"Pre-normalization — X: {X.shape}, y: {y.shape}")
    log.info(f"Class balance — healthy (0): {(y==0).sum()} | depressed (1): {(y==1).sum()}")
    log.info(f"Skipped: {skipped}/{len(uids)} students")

    # ── MinMaxScaler ───────────────────────────────────────────────────────────
    n_samples, n_steps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, n_steps, n_features).astype(np.float32)

    # ── Save ───────────────────────────────────────────────────────────────────
    x_path      = PROC_DIR / "X_train.npy"
    y_path      = PROC_DIR / "y_train.npy"
    scaler_path = PROC_DIR / "scaler.pkl"

    np.save(str(x_path), X_scaled)
    np.save(str(y_path), y)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    log.info(f"Saved X_train.npy → {x_path}  shape: {X_scaled.shape}")
    log.info(f"Saved y_train.npy → {y_path}  shape: {y.shape}")
    log.info(f"Saved scaler.pkl  → {scaler_path}")
    log.info("\n✅ Phase A COMPLETE")


if __name__ == "__main__":
    run_preprocessing()
