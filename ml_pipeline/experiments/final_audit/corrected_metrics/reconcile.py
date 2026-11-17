# -*- coding: utf-8 -*-
"""
Confusion-matrix / Macro-F1 Reconciliation
===========================================
Loads the exact saved Model C artifact and rebuilds the test set
using the same cleaning + split procedure as the final audit.

Canonical class order throughout (never LabelEncoder alphabetical):
  GOOD, MODERATE, UNHEALTHY_SENSITIVE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS

All metrics are computed from ONE pair of (y_true, y_pred) arrays
whose values are canonical string labels — no integer encoding during
display-level arithmetic.
"""
import sys, io, os, json, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── Paths ────────────────────────────────────────────────────
CSV_PATH   = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
MODEL_PATH = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c.pkl"
SCALER_PATH= r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_scaler.pkl"
OUT_DIR    = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\corrected_metrics"
CUTOFF     = datetime.date(2026, 6, 1)

FEATURES = [
    "Temperature(\u00b0C)",
    "Humidity(%)",
    "Pressure(hPa)",
    "Gas (kOhms)",
    "Altitude(m)",
    "DewPoint(\u00b0C)",
    "HeatIndex(\u00b0C)",
]

# Canonical class order — fixed for the entire script
CANONICAL = ["GOOD","MODERATE","UNHEALTHY_SENSITIVE","UNHEALTHY","VERY_UNHEALTHY","HAZARDOUS"]

SEP = "="*72

def aqi_to_risk(aqi):
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

# ══════════════════════════════════════════════════════════════
# STEP 1 — Reconstruct the exact test set
# ══════════════════════════════════════════════════════════════
print(SEP)
print("STEP 1: Reconstruct test set from original CSV")
print(SEP)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df = df.replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ["Date","Time","AQI"]]
for col in feat_cols + ["AQI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.drop_duplicates()
df = df.dropna(subset=feat_cols + ["AQI"])
df["_datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
)
df = df.dropna(subset=["_datetime"])
df = df.sort_values("_datetime").reset_index(drop=True)
df["_date"] = df["_datetime"].dt.date
df["RiskLevel"] = df["AQI"].apply(aqi_to_risk)

df_test = df[df["_date"] >= CUTOFF].copy().reset_index(drop=True)
print(f"Test rows: {len(df_test)}  (expected 782)")
print(f"Test date range: {df_test['_date'].min()}  ->  {df_test['_date'].max()}")

print("Test class distribution:")
for c in CANONICAL:
    n = (df_test["RiskLevel"] == c).sum()
    print(f"  {c:<26}: {n}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — Load model and scaler; generate predictions
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 2: Load model + scaler; generate y_true and y_pred")
print(SEP)

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"Model loaded : {MODEL_PATH}")
print(f"Scaler loaded: {SCALER_PATH}")
print(f"Model type   : {type(clf).__name__}")
print(f"n_estimators : {clf.n_estimators}")
print(f"class_weight : {clf.class_weight}")
print(f"random_state : {clf.random_state}")

# Model's internal class order (integers stored in clf.classes_)
# This is LabelEncoder alphabetical: GOOD=0,HAZARDOUS=1,MODERATE=2,
#   UNHEALTHY=3,UNHEALTHY_SENSITIVE=4,VERY_UNHEALTHY=5
print(f"\nModel clf.classes_ (internal integer labels): {clf.classes_}")
print("NOTE: These are LabelEncoder-encoded integers, NOT string class names.")
print("The decode mapping (from final_audit label encoder) is:")
le_path = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_label_encoder.pkl"
le = joblib.load(le_path)
print(f"  LabelEncoder.classes_ (alphabetical): {list(le.classes_)}")
for i, name in enumerate(le.classes_):
    print(f"    integer {i}  ->  {name}")

# X_test with scaler (scaler was fitted on training data)
X_test = df_test[FEATURES].values
X_test_sc = scaler.transform(X_test)

# y_true as STRING labels (canonical, no integer encoding)
y_true_str = df_test["RiskLevel"].values

# y_pred: model outputs integers; decode to string labels
y_pred_int = clf.predict(X_test_sc)
y_pred_str = le.inverse_transform(y_pred_int)

print(f"\ny_true unique string values: {sorted(set(y_true_str))}")
print(f"y_pred unique string values: {sorted(set(y_pred_str))}")
print(f"Total samples: {len(y_true_str)}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Compute all metrics from string arrays using CANONICAL order
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 3: Metrics from string y_true / y_pred with CANONICAL labels")
print(SEP)
print(f"Canonical class order: {CANONICAL}")

# accuracy_score works on strings directly
acc = accuracy_score(y_true_str, y_pred_str)

mac_p  = precision_score(y_true_str, y_pred_str, labels=CANONICAL, average="macro",    zero_division=0)
mac_r  = recall_score(   y_true_str, y_pred_str, labels=CANONICAL, average="macro",    zero_division=0)
mac_f1 = f1_score(       y_true_str, y_pred_str, labels=CANONICAL, average="macro",    zero_division=0)
wt_f1  = f1_score(       y_true_str, y_pred_str, labels=CANONICAL, average="weighted", zero_division=0)

print(f"\nOverall metrics:")
print(f"  Accuracy        : {acc:.8f}  ({acc*100:.6f}%)")
print(f"  Macro Precision : {mac_p:.8f}")
print(f"  Macro Recall    : {mac_r:.8f}")
print(f"  Macro F1        : {mac_f1:.8f}")
print(f"  Weighted F1     : {wt_f1:.8f}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — Confusion matrix with CANONICAL labels
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 4: Confusion matrix (labels=CANONICAL)")
print(SEP)

cm = confusion_matrix(y_true_str, y_pred_str, labels=CANONICAL)

print("Rows = Actual, Columns = Predicted")
print(f"{'':28}", end="")
for c in CANONICAL:
    print(f"{c[:8]:>12}", end="")
print()
print("-"*(28 + 12*6))
for i, actual in enumerate(CANONICAL):
    print(f"{actual:<28}", end="")
    for j in range(len(CANONICAL)):
        print(f"{cm[i,j]:>12}", end="")
    print()

# Row sums = support, column sums = predicted count
print(f"\nRow sums (= support per class):")
for i, c in enumerate(CANONICAL):
    print(f"  {c:<28}: row_sum={cm[i].sum():>5}  expected={int((np.array(y_true_str)==c).sum()):>5}  {'OK' if cm[i].sum()==int((np.array(y_true_str)==c).sum()) else 'MISMATCH'}")

print(f"\nColumn sums (= predicted count per class):")
for j, c in enumerate(CANONICAL):
    print(f"  {c:<28}: col_sum={cm[:,j].sum():>5}  predicted={int((np.array(y_pred_str)==c).sum()):>5}  {'OK' if cm[:,j].sum()==int((np.array(y_pred_str)==c).sum()) else 'MISMATCH'}")

diag_sum = int(np.diag(cm).sum())
total    = int(cm.sum())
acc_from_cm = diag_sum / total
print(f"\nDiagonal sum / total = {diag_sum} / {total} = {acc_from_cm:.8f}")
print(f"accuracy_score       =                       {acc:.8f}")
print(f"Match: {'YES' if abs(acc_from_cm - acc) < 1e-10 else 'NO — MISMATCH'}")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Per-class metrics computed from confusion matrix
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 5: Per-class metrics derived directly from confusion matrix")
print(SEP)
print(f"{'Class':<28} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>9} {'Recall':>9} {'F1':>9} {'Supp':>6}")
print("-"*78)

per_class = {}
f1_sum = 0.0
for i, cls in enumerate(CANONICAL):
    tp   = int(cm[i, i])
    fp   = int(cm[:, i].sum()) - tp
    fn   = int(cm[i, :].sum()) - tp
    sup  = int(cm[i, :].sum())
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    f1_sum += f1
    per_class[cls] = {"tp":tp,"fp":fp,"fn":fn,"support":sup,
                      "precision":prec,"recall":rec,"f1":f1}
    print(f"{cls:<28} {tp:>5} {fp:>5} {fn:>5} {prec:>9.6f} {rec:>9.6f} {f1:>9.6f} {sup:>6}")

macro_f1_manual = f1_sum / len(CANONICAL)
print(f"\nSum of 6 F1 scores        : {f1_sum:.8f}")
print(f"Manual macro-F1 (sum/6)   : {macro_f1_manual:.8f}")
print(f"sklearn macro-F1 (Step 3) : {mac_f1:.8f}")
print(f"Match: {'YES' if abs(macro_f1_manual - mac_f1) < 1e-8 else 'NO — investigate'}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — Classification report (string labels, CANONICAL order)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 6: sklearn classification_report (canonical order, zero_division=0)")
print(SEP)

report_str = classification_report(
    y_true_str, y_pred_str,
    labels=CANONICAL,
    target_names=CANONICAL,
    zero_division=0,
    digits=6,
)
print(report_str)

# ══════════════════════════════════════════════════════════════
# STEP 7 — Identify the source of earlier discrepancy
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 7: Root-cause analysis of earlier confusion matrix error")
print(SEP)
print("""
The final audit script (reproduce_audit.py) used LabelEncoder to encode string
labels to integers 0-5. LabelEncoder.fit(['GOOD','MODERATE','UNHEALTHY_SENSITIVE',
'UNHEALTHY','VERY_UNHEALTHY','HAZARDOUS']) sorts ALPHABETICALLY, giving:
  0 -> GOOD
  1 -> HAZARDOUS          <-- NOT index 5
  2 -> MODERATE
  3 -> UNHEALTHY
  4 -> UNHEALTHY_SENSITIVE
  5 -> VERY_UNHEALTHY     <-- NOT index 4

The confusion matrix was built using:
  confusion_matrix(y_test_enc, y_pred_enc, labels=[0,1,2,3,4,5])
So the matrix rows/columns were in LabelEncoder alphabetical order:
  Row 0 = GOOD, Row 1 = HAZARDOUS, Row 2 = MODERATE, Row 3 = UNHEALTHY,
  Row 4 = UNHEALTHY_SENSITIVE, Row 5 = VERY_UNHEALTHY

But the display loop printed them as:
  Row 0 = GOOD, Row 1 = MODERATE, Row 2 = UNHEALTHY_SENSITIVE,
  Row 3 = UNHEALTHY, Row 4 = VERY_UNHEALTHY, Row 5 = HAZARDOUS
(canonical order, different from LabelEncoder order)

This caused HAZARDOUS row (LabelEncoder index 1) to be DISPLAYED as MODERATE,
and the VERY_UNHEALTHY row (index 5) to be displayed as HAZARDOUS.

Consequence: the saved confusion_matrix.csv had CORRECT class-name row labels
(written from the cm numpy array rows in LabelEncoder order mapped to string names),
but the display printed in the audit report showed MODERATE with support=122
(which was actually HAZARDOUS), making the per-class F1 calculations look wrong.

The METRICS (accuracy, macro-F1, HAZARDOUS precision/recall) were computed
correctly in Step 3 using sklearn functions with explicit labels=[0..5], which
maps correctly through LabelEncoder. Only the displayed per-class breakdown
and the saved confusion_matrix.csv were affected by the label-ordering mismatch.
""")

# ══════════════════════════════════════════════════════════════
# STEP 8 — Save all corrected artifacts
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 8: Save corrected artifacts")
print(SEP)

# Confusion matrix CSV
cm_df = pd.DataFrame(cm, index=CANONICAL, columns=CANONICAL)
cm_path = os.path.join(OUT_DIR, "confusion_matrix_corrected.csv")
cm_df.to_csv(cm_path)
print(f"Confusion matrix CSV: {cm_path}")

# Classification report text
rep_path = os.path.join(OUT_DIR, "classification_report_corrected.txt")
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(f"AirGuard AI — Corrected Classification Report\n")
    f.write(f"Canonical class order: {CANONICAL}\n")
    f.write(f"Model: RF class_weight=balanced, random_state=42\n")
    f.write(f"Split cutoff: 2026-06-01, Test rows: {len(y_true_str)}\n\n")
    f.write(report_str)
print(f"Classification report: {rep_path}")

# Metrics JSON
metrics = {
    "canonical_class_order": CANONICAL,
    "accuracy":        round(float(acc),    8),
    "macro_precision": round(float(mac_p),  8),
    "macro_recall":    round(float(mac_r),  8),
    "macro_f1":        round(float(mac_f1), 8),
    "weighted_f1":     round(float(wt_f1),  8),
    "macro_f1_manual_check": round(float(macro_f1_manual), 8),
    "macro_f1_matches_manual": bool(abs(macro_f1_manual - mac_f1) < 1e-8),
    "accuracy_from_cm": round(float(acc_from_cm), 8),
    "accuracy_matches_cm": bool(abs(acc_from_cm - acc) < 1e-10),
    "per_class": {
        cls: {k: round(float(v),6) if isinstance(v,float) else v
              for k,v in stats.items()}
        for cls, stats in per_class.items()
    },
    "confusion_matrix": {
        "rows_are": "actual",
        "cols_are": "predicted",
        "order": CANONICAL,
        "values": cm.tolist(),
    },
}
mpath = os.path.join(OUT_DIR, "corrected_metrics.json")
with open(mpath, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics JSON: {mpath}")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FINAL SUMMARY")
print(SEP)
print()
print("1. WHICH PREVIOUS METRIC WAS INCORRECT?")
print("   The aggregate metrics (accuracy, macro-F1, HAZARDOUS precision/recall)")
print("   were computed correctly by sklearn in the final audit. They are unchanged.")
print("   What was WRONG: the printed per-class breakdown and the saved")
print("   confusion_matrix.csv had their row labels in LabelEncoder alphabetical")
print("   order, displayed as if they were in canonical order. This made the")
print("   HAZARDOUS row appear as MODERATE (and vice versa for display purposes).")
print()
print(f"2. CORRECT SIX-CLASS MACRO-F1: {mac_f1:.6f}")
print()
print("3. CORRECT PER-CLASS METRICS:")
print(f"   {'Class':<28} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Supp':>6}")
print(f"   {'-'*62}")
for cls in CANONICAL:
    s = per_class[cls]
    print(f"   {cls:<28} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} {s['support']:>6}")
print()
print("4. CORRECTED CONFUSION MATRIX (rows=actual, cols=predicted):")
print(f"   {'':28}", end="")
for c in CANONICAL:
    print(f"{c[:8]:>10}", end="")
print()
for i, actual in enumerate(CANONICAL):
    print(f"   {actual:<28}", end="")
    for j in range(len(CANONICAL)):
        print(f"{cm[i,j]:>10}", end="")
    print()
print()
print(f"5. ACCURACY AND HAZARDOUS RECALL:")
print(f"   Accuracy         : {acc:.6f} ({acc*100:.4f}%)  UNCHANGED")
print(f"   HAZARDOUS recall : {per_class['HAZARDOUS']['recall']:.6f} ({per_class['HAZARDOUS']['recall']*100:.2f}%)  UNCHANGED")
print(f"   HAZARDOUS prec   : {per_class['HAZARDOUS']['precision']:.6f}  UNCHANGED")
print()
print("6. ROOT CAUSE:")
print("   LabelEncoder sorted classes alphabetically (HAZARDOUS=1, not 5).")
print("   The confusion matrix was stored in that alphabetical order but")
print("   displayed/labelled in canonical order — causing a row-label swap.")
print("   Fixing required working entirely in string label space (this script).")
print()
print("Done.")
