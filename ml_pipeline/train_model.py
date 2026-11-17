"""
AirGuard AI — RandomForest Training Pipeline
=============================================
Dataset : AeroSense Device 2 (Google Sheets export)
Inputs  : 7 sensor features only (AQI is NEVER used as input)
Target  : RiskLevel labels derived from AQI column using
          RiskLevel.fromAqi() thresholds from the Android app
Split   : CHRONOLOGICAL only — random split is forbidden

Disclaimer
----------
Labels are derived from the dataset's own AQI column.
They are NOT independently verified against official EPA AQI
measurements. The dataset's AQI values come from the AeroSense
Device 2 sensor system and may differ from regulatory standards.

RiskLevel thresholds (must match RiskLevel.fromAqi() exactly):
  AQI <= 20  ->  GOOD
  AQI <= 40  ->  MODERATE
  AQI <= 60  ->  UNHEALTHY_SENSITIVE
  AQI <= 80  ->  UNHEALTHY
  AQI <= 90  ->  VERY_UNHEALTHY
  AQI >  90  ->  HAZARDOUS

RandomForestClassifier hyperparameters (all documented below):
  n_estimators      = 200    (number of trees in the forest)
  max_depth         = None   (trees grow to full depth — no pruning)
  min_samples_split = 5      (min samples needed to split an internal node)
  min_samples_leaf  = 2      (min samples required at a leaf node)
  max_features      = "sqrt" (features considered per split = sqrt(7) ~ 2-3)
  class_weight      = "balanced"  (adjusts weights inversely to class frequency)
  random_state      = 42     (reproducibility seed — fix this for all runs)
  n_jobs            = -1     (use all CPU cores)
"""

import sys
import os
import io
import warnings
warnings.filterwarnings("ignore")

# Force stdout/stderr to UTF-8 so Unicode characters don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

# ─────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────

CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))   # ml_pipeline/

MODEL_PATH       = os.path.join(OUT_DIR, "airguard_model.pkl")
SCALER_PATH      = os.path.join(OUT_DIR, "airguard_preprocessor.pkl")
LABEL_ENC_PATH   = os.path.join(OUT_DIR, "airguard_label_encoder.pkl")
FEAT_IMP_PATH    = os.path.join(OUT_DIR, "feature_importances.csv")

FEATURES = [
    "Temperature(°C)",
    "Humidity(%)",
    "Pressure(hPa)",
    "Gas (kOhms)",
    "Altitude(m)",
    "DewPoint(°C)",
    "HeatIndex(°C)",
]
AQI_COL  = "AQI"
DATE_COL = "Date"
TIME_COL = "Time"
TARGET   = "RiskLevel"

TRAIN_RATIO = 0.80   # chronologically earliest 80% → training

RF_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = None,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features      = "sqrt",
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,
)

ALL_CLASSES = [
    "GOOD",
    "MODERATE",
    "UNHEALTHY_SENSITIVE",
    "UNHEALTHY",
    "VERY_UNHEALTHY",
    "HAZARDOUS",
]

# ─────────────────────────────────────────────────────────────
# 1.  Load raw CSV
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  AirGuard AI — ML Training Pipeline")
print("=" * 65)

print(f"\n[1] Loading CSV: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    sys.exit(
        f"\nERROR: CSV file not found at:\n  {CSV_PATH}\n"
        "Please check the path and try again."
    )

df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"    Raw shape : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"    Columns   : {list(df_raw.columns)}")

# ─────────────────────────────────────────────────────────────
# 2.  Verify required columns
# ─────────────────────────────────────────────────────────────
required = FEATURES + [AQI_COL, DATE_COL, TIME_COL]
missing  = [c for c in required if c not in df_raw.columns]
if missing:
    sys.exit(f"\nERROR: Missing required columns: {missing}")
print("\n[2] All required columns present [OK]")

# ─────────────────────────────────────────────────────────────
# 3.  Replace literal "N/A" strings → real NaN
# ─────────────────────────────────────────────────────────────
df = df_raw.replace("N/A", np.nan).copy()

# ─────────────────────────────────────────────────────────────
# 4.  Convert numeric columns (coerce non-numeric → NaN)
# ─────────────────────────────────────────────────────────────
print("\n[3] Cleaning data …")
for col in FEATURES + [AQI_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ─────────────────────────────────────────────────────────────
# 5.  Remove exact duplicates
# ─────────────────────────────────────────────────────────────
n_before_dup = len(df)
df.drop_duplicates(inplace=True)
n_dup = n_before_dup - len(df)
print(f"    Duplicate rows removed                 : {n_dup}")

# ─────────────────────────────────────────────────────────────
# 6.  Drop rows where ANY feature or AQI is NaN
# ─────────────────────────────────────────────────────────────
n_before_na = len(df)
df.dropna(subset=FEATURES + [AQI_COL], inplace=True)
n_na = n_before_na - len(df)
print(f"    Rows dropped (NaN in features or AQI) : {n_na}")
print(f"    Rows remaining after cleaning         : {len(df)}")

if len(df) == 0:
    sys.exit("\nERROR: No rows remain after cleaning. Cannot proceed.")

# ─────────────────────────────────────────────────────────────
# 7.  Derive labels from AQI  (AQI is used ONLY here)
# ─────────────────────────────────────────────────────────────
print("\n[4] Deriving RiskLevel labels from AQI …")
print("    (AQI is used only to build labels — it is excluded from features)")
print("    Disclaimer: labels reflect dataset AQI values, not EPA-verified AQI.\n")

def aqi_to_risk(aqi):
    """
    Mirror of RiskLevel.fromAqi() in the Android app.
    Must stay in sync with any future changes to that function.
    """
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

df[TARGET] = df[AQI_COL].apply(aqi_to_risk)

# AQI statistics (informational only)
print("    AQI statistics (label derivation reference only):")
print(f"      Min    : {df[AQI_COL].min():.2f}")
print(f"      Max    : {df[AQI_COL].max():.2f}")
print(f"      Mean   : {df[AQI_COL].mean():.2f}")
print(f"      Median : {df[AQI_COL].median():.2f}")
print(f"      Std    : {df[AQI_COL].std():.2f}")

# Class distribution over all cleaned rows
print("\n    Full class distribution (all cleaned rows):")
class_counts = df[TARGET].value_counts()
for cls in ALL_CLASSES:
    cnt = class_counts.get(cls, 0)
    pct = cnt / len(df) * 100
    flag = "  [!] ZERO SAMPLES" if cnt == 0 else ""
    print(f"      {cls:<26} : {cnt:>6}  ({pct:5.1f}%){flag}")

zero_classes = [c for c in ALL_CLASSES if class_counts.get(c, 0) == 0]
if zero_classes:
    print(f"\n    [!] WARNING: Classes with zero samples: {zero_classes}")
    print("      These classes cannot be trained or evaluated.")
    print("      This is a DATA LIMITATION of the AeroSense Device 2 dataset.")

# ─────────────────────────────────────────────────────────────
# 8.  Chronological sort & split
# ─────────────────────────────────────────────────────────────
print("\n[5] Chronological split …")

# Separate rows that have parseable datetime from those that do not
mask_has_date = df[DATE_COL].notna() & df[TIME_COL].notna()
df_dated   = df[mask_has_date].copy()
df_no_date = df[~mask_has_date].copy()

print(f"    Rows with date+time available  : {len(df_dated)}")
print(f"    Rows without date/time (NaN)   : {len(df_no_date)}")

if len(df_no_date) > 0:
    print("    → Undated rows cannot be placed on the timeline.")
    print("      They are excluded from training and testing.")
    print("      This avoids contaminating the chronological order.")

# Parse datetime
df_dated["_datetime"] = pd.to_datetime(
    df_dated[DATE_COL].astype(str) + " " + df_dated[TIME_COL].astype(str),
    errors="coerce",
)
n_bad_dt = df_dated["_datetime"].isna().sum()
if n_bad_dt > 0:
    print(f"    Rows with unparseable datetime strings : {n_bad_dt} → excluded")
    df_dated = df_dated.dropna(subset=["_datetime"])

print(f"    Rows usable for chronological split    : {len(df_dated)}")

if len(df_dated) < 10:
    sys.exit(
        "\nERROR: Too few rows with valid datetime to form a meaningful split.\n"
        "Cannot proceed with chronological split. "
        "Please check the Date/Time columns in the CSV."
    )

# Sort by datetime ascending (earliest → latest)
df_dated.sort_values("_datetime", inplace=True)
df_dated.reset_index(drop=True, inplace=True)

date_min = df_dated["_datetime"].min()
date_max = df_dated["_datetime"].max()
print(f"\n    Timeline: {date_min}  →  {date_max}")

# 80/20 split at the chronological boundary
split_idx  = int(len(df_dated) * TRAIN_RATIO)
df_train   = df_dated.iloc[:split_idx].copy()
df_test    = df_dated.iloc[split_idx:].copy()

train_start = df_train["_datetime"].min()
train_end   = df_train["_datetime"].max()
test_start  = df_test["_datetime"].min()
test_end    = df_test["_datetime"].max()

print(f"    Train : {len(df_train):>6} rows  ({train_start}  →  {train_end})")
print(f"    Test  : {len(df_test):>6} rows  ({test_start}  →  {test_end})")

# Class distribution in each split
print("\n    Class distribution — Train split:")
train_counts = df_train[TARGET].value_counts()
for cls in ALL_CLASSES:
    cnt = train_counts.get(cls, 0)
    print(f"      {cls:<26} : {cnt:>6}")

print("\n    Class distribution — Test split:")
test_counts = df_test[TARGET].value_counts()
for cls in ALL_CLASSES:
    cnt = test_counts.get(cls, 0)
    print(f"      {cls:<26} : {cnt:>6}")

# ─────────────────────────────────────────────────────────────
# 9.  Prepare X / y
# ─────────────────────────────────────────────────────────────
X_train = df_train[FEATURES].values
y_train = df_train[TARGET].values
X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values

print(f"\n[6] Feature matrix shapes:")
print(f"    X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"    X_test  : {X_test.shape}    y_test  : {y_test.shape}")

# Feature statistics (from training data only — no data leakage)
print("\n    Feature statistics (training set):")
feat_df = pd.DataFrame(X_train, columns=FEATURES)
print(feat_df.describe().to_string())

# ─────────────────────────────────────────────────────────────
# 10.  Encode labels
# ─────────────────────────────────────────────────────────────
present_classes = sorted(list(set(y_train) | set(y_test)))
le = LabelEncoder()
le.fit(present_classes)
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

# ─────────────────────────────────────────────────────────────
# 11.  Preprocessing (StandardScaler fit on TRAIN only)
# ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)        # transform test (no fit — avoids leakage)

# ─────────────────────────────────────────────────────────────
# 12.  Train RandomForestClassifier
# ─────────────────────────────────────────────────────────────
print("\n[7] Training RandomForestClassifier …")
print(f"    Hyperparameters: {RF_PARAMS}")

clf = RandomForestClassifier(**RF_PARAMS)
clf.fit(X_train_scaled, y_train_enc)
print("    Training complete [OK]")

# ─────────────────────────────────────────────────────────────
# 13.  Evaluate on test set
# ─────────────────────────────────────────────────────────────
print("\n[8] Evaluation on test set (chronologically unseen data):")

y_pred_enc = clf.predict(X_test_scaled)
y_pred     = le.inverse_transform(y_pred_enc)

acc          = accuracy_score(y_test_enc, y_pred_enc)
macro_prec   = precision_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
macro_recall = recall_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
macro_f1     = f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)

print(f"\n    Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
print(f"    Macro Precision   : {macro_prec:.4f}")
print(f"    Macro Recall      : {macro_recall:.4f}")
print(f"    Macro F1-score    : {macro_f1:.4f}")

# ─────────────────────────────────────────────────────────────
# 14.  Majority-class baseline
# ─────────────────────────────────────────────────────────────
majority_class_enc = np.bincount(y_train_enc).argmax()
majority_class     = le.inverse_transform([majority_class_enc])[0]
baseline_pred      = np.full_like(y_test_enc, majority_class_enc)
baseline_acc       = accuracy_score(y_test_enc, baseline_pred)

print(f"\n    Majority-class baseline:")
print(f"      Majority class  : {majority_class}")
print(f"      Baseline accuracy: {baseline_acc:.4f}  ({baseline_acc*100:.2f}%)")
print(f"      Model improvement over baseline: "
      f"{(acc - baseline_acc)*100:+.2f} percentage points")

# ─────────────────────────────────────────────────────────────
# 15.  Classification report (per-class metrics)
# ─────────────────────────────────────────────────────────────
print("\n    Per-class metrics (classification report):")
target_names_present = list(le.classes_)
report = classification_report(
    y_test_enc, y_pred_enc,
    target_names=target_names_present,
    zero_division=0,
)
print(report)

# ─────────────────────────────────────────────────────────────
# 16.  Confusion matrix
# ─────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test_enc, y_pred_enc)
print("    Confusion matrix (rows = actual, cols = predicted):")
print(f"    Classes: {target_names_present}")
header = "         " + "  ".join(f"{c[:6]:>6}" for c in target_names_present)
print(header)
for i, row in enumerate(cm):
    row_str = f"    {target_names_present[i][:6]:>6}  " + "  ".join(f"{v:>6}" for v in row)
    print(row_str)

# ─────────────────────────────────────────────────────────────
# 17.  Feature importances
# ─────────────────────────────────────────────────────────────
print("\n[9] Feature importances (from trained forest):")
importances = clf.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": importances,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(f"    {'Feature':<26}  Importance")
print(f"    {'-'*26}  ----------")
for _, row in feat_imp_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"    {row['Feature']:<26}  {row['Importance']:.6f}  {bar}")

feat_imp_df.to_csv(FEAT_IMP_PATH, index=False)
print(f"\n    Saved to: {FEAT_IMP_PATH}")

# ─────────────────────────────────────────────────────────────
# 18.  Save model and preprocessor
# ─────────────────────────────────────────────────────────────
print("\n[10] Saving model artifacts …")
joblib.dump(clf,    MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le,     LABEL_ENC_PATH)
print(f"    Model       : {MODEL_PATH}")
print(f"    Preprocessor: {SCALER_PATH}")
print(f"    LabelEncoder: {LABEL_ENC_PATH}")

# ─────────────────────────────────────────────────────────────
# 19.  Dataset statistics summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  Raw rows loaded            : {df_raw.shape[0]}")
print(f"  Duplicate rows removed     : {n_dup}")
print(f"  NaN rows removed           : {n_na}")
print(f"  Undated/unparseable rows   : {len(df_no_date) + n_bad_dt}")
print(f"  Rows used for modelling    : {len(df_dated)}")
print(f"  Training rows              : {len(df_train)}")
print(f"  Test rows                  : {len(df_test)}")
print(f"  Split type                 : CHRONOLOGICAL ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})")
print(f"  Features used              : {len(FEATURES)} (AQI is NOT a feature)")
print()
print(f"  Accuracy                   : {acc:.4f}")
print(f"  Macro F1                   : {macro_f1:.4f}")
print(f"  Macro Precision            : {macro_prec:.4f}")
print(f"  Macro Recall               : {macro_recall:.4f}")
print(f"  Majority-class baseline    : {baseline_acc:.4f}")
print()
print("  Limitations:")
if zero_classes:
    print(f"    - Classes never seen in data: {zero_classes}")
    print("      Model cannot predict these classes.")
print("    - Labels are AQI-derived (dataset internal), not EPA-validated.")
print("    - Sensor data collected from one device in one location.")
print("    - Temporal generalization to new dates/environments not tested.")
print("    - StandardScaler parameters must ship alongside the model for")
print("      correct inference on new data.")
print()
print("  Android integration suitability:")
print("    The .pkl files are NOT directly loadable on Android.")
print("    To deploy on Android you must either:")
print("      (a) Convert to ONNX via sklearn-onnx + use ONNX Runtime for Android, OR")
print("      (b) Re-implement the trained forest as hand-coded Kotlin decision trees")
print("          (matching the existing DecisionTree.kt pattern in the app), OR")
print("      (c) Call a hosted REST API serving the model.")
print("    The exported feature_importances.csv and classification metrics are")
print("    immediately usable in a research paper without any Android changes.")
print()
print("=" * 65)
print("  Pipeline complete.")
print("=" * 65)
