# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

CSV_PATH   = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
OUT_DIR    = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments"
CUTOFF     = datetime.date(2026, 6, 1)

FEATURES = [
    "Temperature(\u00b0C)", "Humidity(%)", "Pressure(hPa)",
    "Gas (kOhms)", "Altitude(m)", "DewPoint(\u00b0C)", "HeatIndex(\u00b0C)"
]
ALL_CLASSES = ['GOOD','MODERATE','UNHEALTHY_SENSITIVE','UNHEALTHY','VERY_UNHEALTHY','HAZARDOUS']

RF_BASE = dict(n_estimators=200, max_depth=None, min_samples_split=5,
               min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)

def aqi_to_risk(aqi):
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

# ── Load & clean ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df = df.replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
for col in feat_cols + ['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates()
df = df.dropna(subset=feat_cols + ['AQI'])
df['RiskLevel'] = df['AQI'].apply(aqi_to_risk)
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['_datetime'])
df = df.sort_values('_datetime').reset_index(drop=True)
df['_date'] = df['_datetime'].dt.date

print(f"Total usable rows: {len(df)}")
print(f"Features used: {FEATURES}")

# ── Chronological split on whole calendar days ────────────────
df_train = df[df['_date'] < CUTOFF].copy()
df_test  = df[df['_date'] >= CUTOFF].copy()

# Verify no date overlap
overlap = set(df_train['_date'].unique()) & set(df_test['_date'].unique())
print(f"\nSplit cutoff: {CUTOFF}")
print(f"Train: {len(df_train)} rows  {df_train['_date'].min()} -> {df_train['_date'].max()}")
print(f"Test : {len(df_test)} rows  {df_test['_date'].min()} -> {df_test['_date'].max()}")
print(f"Date overlap: {len(overlap)} [{'CLEAN' if len(overlap)==0 else 'PROBLEM'}]")

print("\nTrain class distribution:")
for c in ALL_CLASSES:
    n = (df_train['RiskLevel']==c).sum()
    print(f"  {c:<26}: {n:>6}  ({n/len(df_train)*100:.2f}%)")

print("\nTest class distribution:")
for c in ALL_CLASSES:
    n = (df_test['RiskLevel']==c).sum()
    print(f"  {c:<26}: {n:>6}  ({n/len(df_test)*100:.2f}%)")

# ── Prepare arrays ────────────────────────────────────────────
X_train = df_train[FEATURES].values
y_train = df_train['RiskLevel'].values
X_test  = df_test[FEATURES].values
y_test  = df_test['RiskLevel'].values

# Encode labels — fit on ALL_CLASSES so encoder is consistent
le = LabelEncoder()
le.fit(ALL_CLASSES)
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

print(f"\nLabelEncoder classes: {list(le.classes_)}")
print(f"Encoded order (index -> class):")
for i, c in enumerate(le.classes_):
    print(f"  {i} -> {c}")

# ── StandardScaler fit on TRAIN only ─────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\nScaler fitted on training data only.")
print("Scaler means:", [round(m,4) for m in scaler.mean_])
print("Scaler scales:", [round(s,4) for s in scaler.scale_])

# ── Helper: evaluate and print ────────────────────────────────
def evaluate(name, y_true, y_pred, le, all_classes):
    # Map encoded preds back to string labels for display
    # Compute metrics in encoded space using all 6 label indices
    all_indices = list(range(len(all_classes)))  # 0..5
    
    acc      = accuracy_score(y_true, y_pred)
    mac_p    = precision_score(y_true, y_pred, labels=all_indices, average='macro', zero_division=0)
    mac_r    = recall_score(y_true, y_pred, labels=all_indices, average='macro', zero_division=0)
    mac_f1   = f1_score(y_true, y_pred, labels=all_indices, average='macro', zero_division=0)
    wt_f1    = f1_score(y_true, y_pred, labels=all_indices, average='weighted', zero_division=0)
    
    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*65}")
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro Precision : {mac_p:.4f}")
    print(f"  Macro Recall    : {mac_r:.4f}")
    print(f"  Macro F1        : {mac_f1:.4f}")
    print(f"  Weighted F1     : {wt_f1:.4f}")
    
    # Per-class metrics
    print(f"\n  Per-class metrics (all 6 classes, zero_division=0):")
    print(f"  {'Class':<26} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Sup':>7}")
    print(f"  {'-'*58}")
    for i, c in enumerate(all_classes):
        idx = le.transform([c])[0] if c in le.classes_ else -1
        sup = int((y_true==idx).sum())
        pred_as = int((y_pred==idx).sum())
        tp  = int(((y_true==idx) & (y_pred==idx)).sum())
        fp  = int(((y_true!=idx) & (y_pred==idx)).sum())
        fn  = int(((y_true==idx) & (y_pred!=idx)).sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        print(f"  {c:<26} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {sup:>7}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(all_classes))))
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    hdr = "         " + "".join(f" {c[:7]:>8}" for c in all_classes)
    print(f"  {hdr}")
    for i, c in enumerate(all_classes):
        row = "".join(f" {v:>8}" for v in cm[i])
        print(f"  {c[:8]:<8} {row}")
    
    # HAZARDOUS check
    haz_idx = le.transform(['HAZARDOUS'])[0]
    haz_tp = int(((y_true==haz_idx) & (y_pred==haz_idx)).sum())
    haz_sup = int((y_true==haz_idx).sum())
    print(f"\n  HAZARDOUS: {haz_tp}/{haz_sup} correctly predicted  (recall={haz_tp/haz_sup:.3f})" if haz_sup>0 else "  HAZARDOUS: no test samples")
    
    return {'name':name,'acc':acc,'mac_p':mac_p,'mac_r':mac_r,'mac_f1':mac_f1,'wt_f1':wt_f1,
            'haz_tp':haz_tp,'haz_sup':haz_sup}

results = []

# ── A. Majority-class baseline ────────────────────────────────
majority_cls_str = df_train['RiskLevel'].value_counts().idxmax()
majority_cls_enc = le.transform([majority_cls_str])[0]
y_base = np.full_like(y_test_enc, majority_cls_enc)
print(f"\nBaseline majority class: {majority_cls_str} (encoded {majority_cls_enc})")
r = evaluate("A. Majority-class Baseline", y_test_enc, y_base, le, ALL_CLASSES)
results.append(r)

# ── B. RF no class weighting ──────────────────────────────────
print("\nTraining RF (no class weight)...")
clf_b = RandomForestClassifier(**RF_BASE, class_weight=None)
clf_b.fit(X_train_sc, y_train_enc)
y_pred_b = clf_b.predict(X_test_sc)
r = evaluate("B. RF class_weight=None", y_test_enc, y_pred_b, le, ALL_CLASSES)
results.append(r)

# ── C. RF balanced ───────────────────────────────────────────
print("\nTraining RF (balanced)...")
clf_c = RandomForestClassifier(**RF_BASE, class_weight='balanced')
clf_c.fit(X_train_sc, y_train_enc)
y_pred_c = clf_c.predict(X_test_sc)
r = evaluate("C. RF class_weight='balanced'", y_test_enc, y_pred_c, le, ALL_CLASSES)
results.append(r)

# ── D. RF balanced_subsample ─────────────────────────────────
print("\nTraining RF (balanced_subsample)...")
clf_d = RandomForestClassifier(**RF_BASE, class_weight='balanced_subsample')
clf_d.fit(X_train_sc, y_train_enc)
y_pred_d = clf_d.predict(X_test_sc)
r = evaluate("D. RF class_weight='balanced_subsample'", y_test_enc, y_pred_d, le, ALL_CLASSES)
results.append(r)

# ── Summary comparison table ──────────────────────────────────
print(f"\n{'='*80}")
print("COMPARISON TABLE (all 6-class metrics, zero_division=0)")
print(f"{'='*80}")
print(f"{'Experiment':<40} {'Acc':>7} {'MacP':>7} {'MacR':>7} {'MacF1':>7} {'WtF1':>7} {'HAZrec':>8}")
print(f"{'-'*80}")
for r in results:
    haz_rec = r['haz_tp']/r['haz_sup'] if r['haz_sup']>0 else 0.0
    print(f"{r['name']:<40} {r['acc']:>7.4f} {r['mac_p']:>7.4f} {r['mac_r']:>7.4f} {r['mac_f1']:>7.4f} {r['wt_f1']:>7.4f} {haz_rec:>8.4f}")

# ── Feature importances for best macro-F1 model ──────────────
best = max(results[1:], key=lambda r: r['mac_f1'])
print(f"\nBest model by macro-F1: {best['name']}")
clf_map = {results[1]['name']: clf_b, results[2]['name']: clf_c, results[3]['name']: clf_d}
best_clf = clf_map[best['name']]
print("Feature importances:")
imps = sorted(zip(FEATURES, best_clf.feature_importances_), key=lambda x: -x[1])
for f, imp in imps:
    bar = '#'*int(imp*50)
    print(f"  {f:<26}: {imp:.6f}  {bar}")
print(f"  Sum: {sum(best_clf.feature_importances_):.6f}")

# ── Save best model artifacts ─────────────────────────────────
import os
joblib.dump(best_clf,  os.path.join(OUT_DIR, 'best_model.pkl'))
joblib.dump(scaler,    os.path.join(OUT_DIR, 'best_preprocessor.pkl'))
joblib.dump(le,        os.path.join(OUT_DIR, 'best_label_encoder.pkl'))

imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': best_clf.feature_importances_})
imp_df = imp_df.sort_values('Importance', ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, 'best_feature_importances.csv'), index=False)
print(f"\nBest model artifacts saved to: {OUT_DIR}")

print("\nDone.")
