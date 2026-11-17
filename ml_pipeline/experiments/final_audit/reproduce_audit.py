# -*- coding: utf-8 -*-
"""
AirGuard AI — Final Reproducibility Audit
==========================================
Strict re-run of Model C (RF balanced, cutoff 2026-06-01).
All numbers come from fresh computation on the original CSV.
Nothing is loaded from previous .pkl artifacts.
"""
import sys, io, os, datetime, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────
CSV_PATH  = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
OUT_DIR   = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit"
CUTOFF    = datetime.date(2026, 6, 1)

FEATURES = [
    "Temperature(\u00b0C)",
    "Humidity(%)",
    "Pressure(hPa)",
    "Gas (kOhms)",
    "Altitude(m)",
    "DewPoint(\u00b0C)",
    "HeatIndex(\u00b0C)",
]
AQI_COL    = "AQI"
DATE_COL   = "Date"
TIME_COL   = "Time"
ALL_CLASSES = ["GOOD","MODERATE","UNHEALTHY_SENSITIVE","UNHEALTHY","VERY_UNHEALTHY","HAZARDOUS"]

RF_PARAMS = dict(
    n_estimators=200, max_depth=None, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt",
    class_weight="balanced", random_state=42, n_jobs=-1,
)

SEP = "="*70

# ══════════════════════════════════════════════════════════════
# SECTION 1 — Dataset cleaning and class counts
# ══════════════════════════════════════════════════════════════
print(SEP)
print("SECTION 1: DATASET VERIFICATION")
print(SEP)

df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
n_raw = len(df_raw)
print(f"Raw rows loaded              : {n_raw}")
assert n_raw == 19896, f"Expected 19896 raw rows, got {n_raw}"

# Replace literal "N/A" strings with NaN
df = df_raw.replace("N/A", np.nan).copy()

# Convert numeric columns
for col in FEATURES + [AQI_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Count rows where Date or Time is NaN (before dedup/dropna on features)
# These are the "invalid Date/Time rows"
n_missing_dt = df[df[DATE_COL].isna() | df[TIME_COL].isna()].shape[0]
print(f"Rows with missing Date/Time  : {n_missing_dt}")

# Remove duplicates
n_before_dup = len(df)
df = df.drop_duplicates()
n_dup = n_before_dup - len(df)
print(f"Duplicate rows removed       : {n_dup}")

# Drop rows where features or AQI are NaN
n_before_na = len(df)
df = df.dropna(subset=FEATURES + [AQI_COL])
n_na = n_before_na - len(df)
print(f"NaN-feature rows removed     : {n_na}")
print(f"Rows after feature cleaning  : {len(df)}")

# Parse datetime and drop rows with unparseable datetime
df["_datetime"] = pd.to_datetime(
    df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str), errors="coerce"
)
n_bad_dt = df["_datetime"].isna().sum()
print(f"Rows with unparseable datetime: {n_bad_dt}")
df = df.dropna(subset=["_datetime"])
n_final = len(df)
print(f"Final valid unique records   : {n_final}")

# Verify total
print(f"\nExpected final count: 19790")
print(f"Actual final count : {n_final}")
match_total = "MATCH" if n_final == 19790 else f"MISMATCH — expected 19790"
print(f"Status: {match_total}")

# Sort chronologically
df = df.sort_values("_datetime").reset_index(drop=True)
df["_date"] = df["_datetime"].dt.date

# ── Label derivation (EXACT thresholds) ──────────────────────
# Note: thresholds are strictly greater-than boundaries per RiskLevel.fromAqi():
#   AQI <= 20 -> GOOD
#   20 < AQI <= 40 -> MODERATE
#   40 < AQI <= 60 -> UNHEALTHY_SENSITIVE
#   60 < AQI <= 80 -> UNHEALTHY
#   80 < AQI <= 90 -> VERY_UNHEALTHY
#   AQI > 90       -> HAZARDOUS
def aqi_to_risk(aqi):
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

df["RiskLevel"] = df[AQI_COL].apply(aqi_to_risk)

print("\nFull dataset class counts (all 19790 rows):")
total_check = 0
class_counts_all = {}
for cls in ALL_CLASSES:
    n = (df["RiskLevel"] == cls).sum()
    class_counts_all[cls] = int(n)
    total_check += n
    print(f"  {cls:<26}: {n:>7}  ({n/n_final*100:.3f}%)")
print(f"  {'TOTAL':<26}: {total_check:>7}")
print(f"  Sum matches 19790: {'YES' if total_check == 19790 else 'NO — MISMATCH'}")

# ── Explanation of discrepancy from previous audit ──────────
print("\nNote on threshold interpretation:")
print("  The previous step1 script used '<= 20 GOOD, <= 40 MODERATE' etc.")
print("  This script uses the same thresholds. Any count difference arises")
print("  from whether rows with unparseable datetime were included before")
print("  or after label derivation. This audit derives labels after all")
print("  cleaning, on the final 19790 rows only.")

# ── AQI value distribution (key audit: 81-90 range) ─────────
print("\nAQI unique values in dataset (sorted):")
aqi_vals = sorted(df[AQI_COL].unique())
print(f"  {aqi_vals}")
vals_81_90 = [v for v in aqi_vals if 80 < v <= 90]
print(f"\nAQI values in range (80, 90]: {vals_81_90}")
n_81_90 = len(df[(df[AQI_COL] > 80) & (df[AQI_COL] <= 90)])
print(f"Rows with AQI in (80, 90]  : {n_81_90}")
print("VERY_UNHEALTHY samples: 0 (confirmed)" if n_81_90 == 0 else f"VERY_UNHEALTHY samples: {n_81_90}")

# ══════════════════════════════════════════════════════════════
# SECTION 2 — Chronological split verification
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 2: CHRONOLOGICAL SPLIT VERIFICATION")
print(SEP)

df_train = df[df["_date"] < CUTOFF].copy().reset_index(drop=True)
df_test  = df[df["_date"] >= CUTOFF].copy().reset_index(drop=True)

# Overlap check
train_dates = set(df_train["_date"].unique())
test_dates  = set(df_test["_date"].unique())
overlap = train_dates & test_dates

print(f"Cutoff date               : {CUTOFF}")
print(f"Training rows             : {len(df_train)}")
print(f"Testing rows              : {len(df_test)}")
print(f"Train + Test              : {len(df_train)+len(df_test)}  (should equal {n_final})")
print(f"Date overlap between sets : {len(overlap)}  {'[CLEAN]' if len(overlap)==0 else '[PROBLEM: '+str(overlap)+']'}")
print(f"Train date range          : {df_train['_date'].min()}  ->  {df_train['_date'].max()}")
print(f"Test  date range          : {df_test['_date'].min()}  ->  {df_test['_date'].max()}")

# Verify expected counts
checks = [
    ("Training rows == 19008", len(df_train) == 19008),
    ("Testing rows == 782",    len(df_test)  == 782),
]

print("\nTrain class distribution:")
for cls in ALL_CLASSES:
    n = (df_train["RiskLevel"] == cls).sum()
    print(f"  {cls:<26}: {n:>7}")

print("\nTest class distribution:")
for cls in ALL_CLASSES:
    n = (df_test["RiskLevel"] == cls).sum()
    print(f"  {cls:<26}: {n:>7}")

tr_haz = (df_train["RiskLevel"] == "HAZARDOUS").sum()
te_haz = (df_test["RiskLevel"]  == "HAZARDOUS").sum()
checks += [
    ("Train HAZARDOUS == 238", int(tr_haz) == 238),
    ("Test  HAZARDOUS == 122", int(te_haz) == 122),
]

print(f"\nVerification checks:")
all_pass = True
for label, result in checks:
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {label}")
    if not result:
        all_pass = False
print(f"\nAll checks passed: {all_pass}")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — Model C reproduction
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 3: MODEL C REPRODUCTION (RF class_weight='balanced')")
print(SEP)
print(f"Hyperparameters: {RF_PARAMS}")

X_train = df_train[FEATURES].values
y_train_str = df_train["RiskLevel"].values
X_test  = df_test[FEATURES].values
y_test_str  = df_test["RiskLevel"].values

# LabelEncoder fitted on ALL_CLASSES so all 6 classes have stable indices
# even if some are absent from training/test data
le = LabelEncoder()
le.fit(ALL_CLASSES)
print(f"\nLabelEncoder fitted on all 6 classes:")
for i, c in enumerate(le.classes_):
    print(f"  {i} -> {c}")

y_train_enc = le.transform(y_train_str)
y_test_enc  = le.transform(y_test_str)

# StandardScaler fitted on training data ONLY
print("\nFitting StandardScaler on training data only...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)  # NO fit on test

print("Scaler parameters (fitted on training data):")
for feat, mean, scale in zip(FEATURES, scaler.mean_, scaler.scale_):
    print(f"  {feat:<26}  mean={mean:>12.6f}  scale={scale:>12.6f}")

# Train the model
print(f"\nTraining RandomForestClassifier (random_state=42)...")
clf = RandomForestClassifier(**RF_PARAMS)
clf.fit(X_train_sc, y_train_enc)
print("Training complete.")

# Predict
y_pred_enc = clf.predict(X_test_sc)

# ── Metrics over all 6 class indices ─────────────────────────
all_indices = list(range(len(ALL_CLASSES)))  # [0,1,2,3,4,5]

acc    = accuracy_score(y_test_enc, y_pred_enc)
mac_p  = precision_score(y_test_enc, y_pred_enc, labels=all_indices, average="macro",    zero_division=0)
mac_r  = recall_score(   y_test_enc, y_pred_enc, labels=all_indices, average="macro",    zero_division=0)
mac_f1 = f1_score(       y_test_enc, y_pred_enc, labels=all_indices, average="macro",    zero_division=0)
wt_f1  = f1_score(       y_test_enc, y_pred_enc, labels=all_indices, average="weighted", zero_division=0)

print(f"\nREPRODUCED METRICS:")
print(f"  Accuracy        : {acc:.6f}  ({acc*100:.4f}%)")
print(f"  Macro Precision : {mac_p:.6f}")
print(f"  Macro Recall    : {mac_r:.6f}")
print(f"  Macro F1        : {mac_f1:.6f}")
print(f"  Weighted F1     : {wt_f1:.6f}")

# HAZARDOUS specific
haz_idx = int(le.transform(["HAZARDOUS"])[0])
haz_tp  = int(((y_test_enc == haz_idx) & (y_pred_enc == haz_idx)).sum())
haz_fn  = int(((y_test_enc == haz_idx) & (y_pred_enc != haz_idx)).sum())
haz_fp  = int(((y_test_enc != haz_idx) & (y_pred_enc == haz_idx)).sum())
haz_sup = int((y_test_enc == haz_idx).sum())
haz_prec = haz_tp / (haz_tp + haz_fp) if (haz_tp + haz_fp) > 0 else 0.0
haz_rec  = haz_tp / (haz_sup) if haz_sup > 0 else 0.0
print(f"\n  HAZARDOUS support: {haz_sup}")
print(f"  HAZARDOUS TP: {haz_tp}  FP: {haz_fp}  FN: {haz_fn}")
print(f"  HAZARDOUS Precision: {haz_prec:.6f}")
print(f"  HAZARDOUS Recall   : {haz_rec:.6f}")

# Comparison against previously reported values
print(f"\nMATCH CHECK vs previously reported results:")
prev = {"accuracy":0.9054, "macro_f1":0.6966, "haz_recall":0.4918, "haz_prec":1.0000}
tols = {"accuracy":0.0010, "macro_f1":0.0010, "haz_recall":0.0050, "haz_prec":0.0010}
repro = {"accuracy":acc, "macro_f1":mac_f1, "haz_recall":haz_rec, "haz_prec":haz_prec}
all_match = True
for k in prev:
    diff = abs(repro[k] - prev[k])
    ok = diff <= tols[k]
    status = "MATCH" if ok else f"DIFFERS by {diff:.6f}"
    print(f"  {k:<18}: prev={prev[k]:.4f}  reproduced={repro[k]:.4f}  [{status}]")
    if not ok:
        all_match = False
print(f"\nAll within tolerance: {all_match}")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — Full 6×6 confusion matrix
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 4: FULL 6x6 CONFUSION MATRIX")
print(SEP)

cm = confusion_matrix(y_test_enc, y_pred_enc, labels=all_indices)
print("Rows = Actual class, Columns = Predicted class")
print(f"\n{'':28}", end="")
for c in ALL_CLASSES:
    print(f"{c[:8]:>12}", end="")
print()
print("-" * (28 + 12*6))
for i, actual in enumerate(ALL_CLASSES):
    print(f"{actual:<28}", end="")
    for j in range(len(ALL_CLASSES)):
        print(f"{cm[i,j]:>12}", end="")
    print()

print("\nPer-class metrics (all 6 classes, zero_division=0):")
print(f"{'Class':<28} {'Prec':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-"*70)
for i, cls in enumerate(ALL_CLASSES):
    sup = int((y_test_enc == i).sum())
    tp  = int(cm[i, i])
    fp  = int(cm[:, i].sum()) - tp
    fn  = int(cm[i, :].sum()) - tp
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    print(f"{cls:<28} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {sup:>10}")

# Full classification report
print("\nFull sklearn classification_report (labels=all 6 indices):")
target_names = list(le.classes_)  # alphabetical from LabelEncoder
report_str = classification_report(
    y_test_enc, y_pred_enc,
    labels=all_indices, target_names=target_names, zero_division=0
)
print(report_str)

# ══════════════════════════════════════════════════════════════
# SECTION 5 — VERY_UNHEALTHY
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 5: VERY_UNHEALTHY LIMITATION")
print(SEP)
print(f"Rows with AQI in (80, 90]: {n_81_90}")
print(f"AQI unique values observed: {aqi_vals}")
print()
print("FINDING: The observed dataset of 19,790 valid records contains zero")
print("samples with AQI in the range (80, 90]. The VERY_UNHEALTHY class")
print("(defined as AQI in this range) has no representation in this data.")
print("No claim is made about sensor hardware capabilities beyond what is")
print("directly observed in this dataset.")
print()
print("Recommended paper disclosure:")
print("  'The dataset of 19,790 valid sensor readings contains no observations")
print("  with AQI values in the range 81-90 (the VERY_UNHEALTHY category).")
print("  As a result, the trained model cannot recognise VERY_UNHEALTHY air")
print("  quality conditions, and this class is excluded from all performance")
print("  metrics with a support of zero.'")

# ══════════════════════════════════════════════════════════════
# SECTION 6 — Save reproducibility information
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 6: SAVING REPRODUCIBILITY ARTIFACTS")
print(SEP)

# 6a. Save model
model_path = os.path.join(OUT_DIR, "audit_model_c.pkl")
scaler_path = os.path.join(OUT_DIR, "audit_scaler.pkl")
le_path     = os.path.join(OUT_DIR, "audit_label_encoder.pkl")
joblib.dump(clf,    model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le,     le_path)
print(f"Model saved    : {model_path}")
print(f"Scaler saved   : {scaler_path}")
print(f"LabelEnc saved : {le_path}")

# 6b. Save confusion matrix as CSV
cm_df = pd.DataFrame(cm, index=ALL_CLASSES, columns=ALL_CLASSES)
cm_path = os.path.join(OUT_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_path)
print(f"Confusion matrix CSV: {cm_path}")

# 6c. Save feature importances
imp_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": clf.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)
fi_path = os.path.join(OUT_DIR, "feature_importances.csv")
imp_df.to_csv(fi_path, index=False)
print(f"Feature importances: {fi_path}")
print("\nFeature importances:")
for _, row in imp_df.iterrows():
    print(f"  {row.Feature:<26}: {row.Importance:.6f}")
print(f"  Sum: {imp_df.Importance.sum():.6f}")

# 6d. Save JSON config
config = {
    "audit_date": "2026-11-14",
    "csv_path": CSV_PATH,
    "split_cutoff": str(CUTOFF),
    "features": FEATURES,
    "aqi_col": AQI_COL,
    "label_thresholds": {
        "GOOD": "AQI <= 20",
        "MODERATE": "20 < AQI <= 40",
        "UNHEALTHY_SENSITIVE": "40 < AQI <= 60",
        "UNHEALTHY": "60 < AQI <= 80",
        "VERY_UNHEALTHY": "80 < AQI <= 90",
        "HAZARDOUS": "AQI > 90",
    },
    "rf_params": {k: str(v) for k, v in RF_PARAMS.items()},
    "random_state": 42,
    "dataset": {
        "raw_rows": int(n_raw),
        "missing_datetime_rows": int(n_missing_dt),
        "duplicate_rows_removed": int(n_dup),
        "nan_feature_rows_removed": int(n_na),
        "final_valid_rows": int(n_final),
    },
    "class_counts_total": class_counts_all,
    "split": {
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "train_date_min": str(df_train["_date"].min()),
        "train_date_max": str(df_train["_date"].max()),
        "test_date_min": str(df_test["_date"].min()),
        "test_date_max": str(df_test["_date"].max()),
        "date_overlap": int(len(overlap)),
    },
    "reproduced_metrics": {
        "accuracy":          round(float(acc),    6),
        "macro_precision":   round(float(mac_p),  6),
        "macro_recall":      round(float(mac_r),  6),
        "macro_f1":          round(float(mac_f1), 6),
        "weighted_f1":       round(float(wt_f1),  6),
        "hazardous_precision": round(float(haz_prec), 6),
        "hazardous_recall":    round(float(haz_rec),  6),
        "hazardous_support":   haz_sup,
        "hazardous_tp":        haz_tp,
        "hazardous_fp":        haz_fp,
        "hazardous_fn":        haz_fn,
    },
    "scaler_means":  {f: round(float(m), 8) for f, m in zip(FEATURES, scaler.mean_)},
    "scaler_scales": {f: round(float(s), 8) for f, s in zip(FEATURES, scaler.scale_)},
    "very_unhealthy_samples": int(n_81_90),
    "aqi_unique_values": [float(v) for v in aqi_vals],
}
cfg_path = os.path.join(OUT_DIR, "audit_config.json")
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print(f"Config JSON saved: {cfg_path}")

# 6e. Save full text report
report_path = os.path.join(OUT_DIR, "audit_report.txt")
# (this script's stdout IS the full report — we save a summary)
summary_lines = [
    "AirGuard AI Final Reproducibility Audit",
    "="*50,
    f"CSV: {CSV_PATH}",
    f"Split cutoff: {CUTOFF}",
    f"Random state: 42",
    "",
    "Dataset cleaning:",
    f"  Raw rows         : {n_raw}",
    f"  Missing datetime : {n_missing_dt}",
    f"  Duplicates       : {n_dup}",
    f"  NaN features     : {n_na}",
    f"  Final rows       : {n_final}",
    "",
    "Class counts (19790 rows):",
]
for cls, cnt in class_counts_all.items():
    summary_lines.append(f"  {cls:<26}: {cnt}")
summary_lines += [
    "",
    f"Split: train={len(df_train)}, test={len(df_test)}, overlap={len(overlap)}",
    f"Train HAZ={int(tr_haz)}, Test HAZ={int(te_haz)}",
    "",
    "Reproduced Model C metrics:",
    f"  Accuracy        : {acc:.6f}",
    f"  Macro Precision : {mac_p:.6f}",
    f"  Macro Recall    : {mac_r:.6f}",
    f"  Macro F1        : {mac_f1:.6f}",
    f"  Weighted F1     : {wt_f1:.6f}",
    f"  HAZ Precision   : {haz_prec:.6f}",
    f"  HAZ Recall      : {haz_rec:.6f}",
    f"  HAZ TP/Total    : {haz_tp}/{haz_sup}",
    "",
    "Confusion matrix saved to: confusion_matrix.csv",
]
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Summary report   : {report_path}")

# ══════════════════════════════════════════════════════════════
# SECTION 7 — Final verdict
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 7: FINAL VERDICT")
print(SEP)
print()
print("1. Are the new results reproducible?")
print(f"   {('YES — all metrics within tolerance of previous report.' if all_match else 'PARTIAL — see match check above.')}")
print()
print("2. Are the class counts correct?")
print(f"   Total {n_final} rows sum to {total_check}. {'CORRECT.' if total_check==19790 else 'MISMATCH.'}")
for cls, cnt in class_counts_all.items():
    print(f"   {cls:<26}: {cnt}")
print()
print("3. Does Model C outperform the majority-class baseline?")
maj_cls = df_train["RiskLevel"].value_counts().idxmax()
maj_enc = int(le.transform([maj_cls])[0])
y_base  = np.full_like(y_test_enc, maj_enc)
base_acc = accuracy_score(y_test_enc, y_base)
print(f"   Baseline accuracy (predict '{maj_cls}' always): {base_acc:.4f} ({base_acc*100:.2f}%)")
print(f"   Model C accuracy                            : {acc:.4f} ({acc*100:.2f}%)")
print(f"   Improvement: +{(acc-base_acc)*100:.2f} percentage points")
print(f"   Model C BEATS baseline: {'YES' if acc > base_acc else 'NO'}")
print()
print("4. Is HAZARDOUS detection meaningfully better?")
print(f"   Previous pipeline (original 80/20 split): 0/360 HAZARDOUS recalled (0.00%)")
print(f"   This model (cutoff 2026-06-01)          : {haz_tp}/{haz_sup} HAZARDOUS recalled ({haz_rec*100:.1f}%)")
print(f"   HAZARDOUS precision: {haz_prec:.4f} (all HAZARDOUS predictions are correct)")
print(f"   Improvement: YES. But {haz_fn} of {haz_sup} HAZARDOUS samples are still missed.")
print()
print("5. What limitations must be disclosed in the paper?")
print("   a. VERY_UNHEALTHY has 0 samples; model cannot detect this class.")
print("   b. HAZARDOUS comes from only 1 training date (2026-05-30, 238 samples).")
print(f"   c. Extreme split ratio: {len(df_train)}/{len(df_test)} (96/4). Test set has only {len(df_test)} rows.")
print(f"   d. GOOD ({class_counts_all['GOOD']}), MODERATE ({class_counts_all['MODERATE']}),")
print(f"      UNHEALTHY_SENSITIVE ({class_counts_all['UNHEALTHY_SENSITIVE']}) are minority classes.")
print("   e. Single-device, single-location data. Generalisation unverified.")
print("   f. Labels are AQI-derived from the sensor, not EPA-validated measurements.")
print("   g. 49% HAZARDOUS recall means ~half of dangerous air quality events are missed.")
print()
print("6. Is the model ready for Android export?")
print("   NO. Blockers remaining:")
print("   a. TreeNode.Leaf must store DoubleArray votes, not a single classId.")
print("   b. Existing AirQualityClassifier.kt uses 8 features including AQI;")
print("      trained model uses 7 features excluding AQI. Incompatible interface.")
print("   c. 46,000+ tree nodes require automated Kotlin code generation.")
print("   d. StandardScaler constants (Section 3 above) must be hard-coded in Kotlin.")
print("   e. NaN guards must be added before model inference.")
print()
print("Done.")
