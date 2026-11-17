# -*- coding: utf-8 -*-
"""
Pre-integration checks A and B:
  A. Model artifact consistency
  B. Full 782-sample sklearn vs ONNX equivalence
"""
import sys, io, os, json, datetime, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib

BASE = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments"
AUDIT = os.path.join(BASE, "final_audit")
CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
FEATURES = [
    "Temperature(\u00b0C)","Humidity(%)","Pressure(hPa)",
    "Gas (kOhms)","Altitude(m)","DewPoint(\u00b0C)","HeatIndex(\u00b0C)",
]
CANONICAL = ["GOOD","MODERATE","UNHEALTHY_SENSITIVE","UNHEALTHY","VERY_UNHEALTHY","HAZARDOUS"]
CUTOFF = datetime.date(2026, 6, 1)
SEP = "="*72

def file_md5(path):
    h = hashlib.md5()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def aqi_to_risk(aqi):
    if aqi<=20: return "GOOD"
    elif aqi<=40: return "MODERATE"
    elif aqi<=60: return "UNHEALTHY_SENSITIVE"
    elif aqi<=80: return "UNHEALTHY"
    elif aqi<=90: return "VERY_UNHEALTHY"
    else: return "HAZARDOUS"

# ══════════════════════════════════════════════════════════════
# CHECK A — Artifact consistency
# ══════════════════════════════════════════════════════════════
print(SEP)
print("CHECK A: MODEL ARTIFACT CONSISTENCY")
print(SEP)

# Paths
best_path   = os.path.join(BASE, "best_model.pkl")
audit_path  = os.path.join(AUDIT, "audit_model_c.pkl")
onnx_path   = os.path.join(AUDIT, "audit_model_c.onnx")
scaler_path = os.path.join(AUDIT, "audit_scaler.pkl")
le_path     = os.path.join(AUDIT, "audit_label_encoder.pkl")
cfg_path    = os.path.join(AUDIT, "audit_config.json")

for label, path in [
    ("best_model.pkl",      best_path),
    ("audit_model_c.pkl",   audit_path),
    ("audit_model_c.onnx",  onnx_path),
    ("audit_scaler.pkl",    scaler_path),
    ("audit_label_encoder.pkl", le_path),
    ("audit_config.json",   cfg_path),
]:
    exists = os.path.exists(path)
    size = os.path.getsize(path)/1024/1024 if exists else 0
    md5  = file_md5(path) if exists else "MISSING"
    print(f"  {label:<30}: {'EXISTS' if exists else 'MISSING':8}  {size:7.2f} MB  MD5={md5[:12]}...")

# Load both models and compare parameters
print("\nComparing best_model.pkl vs audit_model_c.pkl:")
clf_best  = joblib.load(best_path)
clf_audit = joblib.load(audit_path)
le = joblib.load(le_path)
scaler = joblib.load(scaler_path)

params_match = True
checks = [
    ("n_estimators",   clf_best.n_estimators,  clf_audit.n_estimators),
    ("n_features_in_", clf_best.n_features_in_, clf_audit.n_features_in_),
    ("n_classes_",     clf_best.n_classes_,     clf_audit.n_classes_),
    ("class_weight",   str(clf_best.class_weight), str(clf_audit.class_weight)),
    ("random_state",   clf_best.random_state,   clf_audit.random_state),
    ("max_features",   clf_best.max_features,   clf_audit.max_features),
]
for name, v_best, v_audit in checks:
    match = v_best == v_audit
    if not match: params_match = False
    print(f"  {name:<20}: best={v_best}  audit={v_audit}  {'MATCH' if match else 'DIFFER'}")

# Compare feature importances (direct indicator of same training)
fi_diff = np.abs(clf_best.feature_importances_ - clf_audit.feature_importances_).max()
print(f"  Max feature importance diff: {fi_diff:.10f}  {'IDENTICAL' if fi_diff < 1e-10 else 'DIFFER'}")
if fi_diff > 1e-10:
    params_match = False

# Compare tree structure of first tree
t_best  = clf_best.estimators_[0].tree_
t_audit = clf_audit.estimators_[0].tree_
same_structure = (
    t_best.node_count == t_audit.node_count and
    np.allclose(t_best.threshold, t_audit.threshold) and
    np.allclose(t_best.value,     t_audit.value)
)
print(f"  Tree 0 structure identical: {same_structure}")
if not same_structure: params_match = False

print(f"\n  VERDICT: Models are {'IDENTICAL' if params_match else 'DIFFERENT'}")
if not params_match:
    print("  WARNING: best_model.pkl and audit_model_c.pkl differ — use audit_model_c.pkl")

# Verify scaler belongs to this experiment (check means match config)
with open(cfg_path) as f:
    cfg = json.load(f)
cfg_means  = [cfg["scaler_means"][feat] for feat in FEATURES]
cfg_scales = [cfg["scaler_scales"][feat] for feat in FEATURES]
mean_diff  = np.abs(np.array(cfg_means)  - scaler.mean_).max()
scale_diff = np.abs(np.array(cfg_scales) - scaler.scale_).max()
print(f"\nScaler consistency (vs audit_config.json):")
print(f"  Max mean diff  : {mean_diff:.2e}  {'OK' if mean_diff<1e-6 else 'MISMATCH'}")
print(f"  Max scale diff : {scale_diff:.2e}  {'OK' if scale_diff<1e-6 else 'MISMATCH'}")

# LabelEncoder
print(f"\nLabelEncoder classes: {list(le.classes_)}")
print(f"  VERY_UNHEALTHY in le.classes_: {'VERY_UNHEALTHY' in le.classes_}")

# ONNX inspection
import onnx
onnx_model = onnx.load(onnx_path)
print(f"\nONNX model inspection:")
print(f"  IR version    : {onnx_model.ir_version}")
print(f"  opset version : {onnx_model.opset_import[0].version}")
inputs = onnx_model.graph.input
outputs = onnx_model.graph.output
print(f"  Inputs ({len(inputs)}):")
for inp in inputs:
    print(f"    {inp.name}: {inp.type.tensor_type.shape}")
print(f"  Outputs ({len(outputs)}):")
for out in outputs:
    print(f"    {out.name}: type={out.type}")

# Verify ONNX feature order by checking metadata
import onnxruntime as rt
sess = rt.InferenceSession(onnx_path)
in_name  = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
print(f"\n  ONNX input name  : {in_name}")
print(f"  ONNX output names: {out_names}")
print(f"  ONNX input shape : {sess.get_inputs()[0].shape}")
print(f"  ONNX output 0 (label) type: {sess.get_outputs()[0].type}")
print(f"  ONNX output 1 (proba) type: {sess.get_outputs()[1].type}")

# Run one sample through ONNX to see output class encoding
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig').replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
for col in feat_cols+['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates().dropna(subset=feat_cols+['AQI'])
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['_datetime']).sort_values('_datetime').reset_index(drop=True)
df['_date'] = df['_datetime'].dt.date
df['RiskLevel'] = df['AQI'].apply(aqi_to_risk)
df_test = df[df['_date'] >= CUTOFF].copy().reset_index(drop=True)
X_test = df_test[FEATURES].values
X_test_sc = scaler.transform(X_test)

x1 = X_test_sc[0:1].astype(np.float32)
onnx_label_raw = sess.run([out_names[0]], {in_name: x1})[0]
onnx_proba_raw = sess.run([out_names[1]], {in_name: x1})[0]
print(f"\n  ONNX sample output — raw label: {onnx_label_raw}  type: {type(onnx_label_raw[0])}")
print(f"  ONNX sample proba dict: {onnx_proba_raw}")

# ONNX outputs a list of dicts for proba; check structure
if isinstance(onnx_proba_raw, list) and len(onnx_proba_raw) > 0:
    proba_dict = onnx_proba_raw[0]
    print(f"  ONNX proba keys (class integers): {sorted(proba_dict.keys())}")
    print(f"  Mapping (via LabelEncoder):")
    for k in sorted(proba_dict.keys()):
        cls = le.inverse_transform([k])[0]
        print(f"    ONNX key {k} -> {cls}  prob={proba_dict[k]:.6f}")

print(f"\nCheck A complete.")

# ══════════════════════════════════════════════════════════════
# CHECK B — Full 782-sample equivalence
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CHECK B: FULL 782-SAMPLE SKLEARN vs ONNX EQUIVALENCE")
print(SEP)

# sklearn predictions (string labels)
y_sk_int = clf_audit.predict(X_test_sc)
y_sk_str = le.inverse_transform(y_sk_int)
sk_proba  = clf_audit.predict_proba(X_test_sc)   # shape (782, 5)

# ONNX predictions
X_test_f32 = X_test_sc.astype(np.float32)
onnx_labels_raw = sess.run([out_names[0]], {in_name: X_test_f32})[0]  # int array
onnx_proba_raw  = sess.run([out_names[1]], {in_name: X_test_f32})[0]  # list of dicts

# Decode ONNX labels to strings
y_onnx_str = le.inverse_transform(onnx_labels_raw)

# Prediction agreement
n_total = len(y_sk_str)
n_match = int((y_sk_str == y_onnx_str).sum())
n_mismatch = n_total - n_match

print(f"Total test samples           : {n_total}")
print(f"Predicted-class matches      : {n_match}")
print(f"Predicted-class mismatches   : {n_mismatch}")
print(f"Agreement rate               : {n_match/n_total*100:.4f}%")
print(f"100% agreement: {'YES' if n_mismatch == 0 else 'NO'}")

if n_mismatch > 0:
    print("\nMismatch details:")
    mismatch_idx = np.where(y_sk_str != y_onnx_str)[0]
    for i in mismatch_idx[:10]:
        print(f"  sample {i}: sklearn={y_sk_str[i]}  onnx={y_onnx_str[i]}")

# Probability comparison
# ONNX proba is list of dicts {int_class: prob}; build (n, 5) array
onnx_proba_arr = np.zeros((n_total, clf_audit.n_classes_), dtype=np.float32)
for i, d in enumerate(onnx_proba_raw):
    for cls_int, prob in d.items():
        idx = list(clf_audit.classes_).index(cls_int)
        onnx_proba_arr[i, idx] = prob

proba_diff = np.abs(sk_proba - onnx_proba_arr)
max_abs_diff = float(proba_diff.max())
mean_abs_diff = float(proba_diff.mean())
per_class_max = proba_diff.max(axis=0)

print(f"\nProbability comparison (sklearn float64 vs ONNX float32):")
print(f"  Max absolute difference (any class, any sample): {max_abs_diff:.2e}")
print(f"  Mean absolute difference                       : {mean_abs_diff:.2e}")
print(f"  Per-class max difference:")
for i, cls_int in enumerate(clf_audit.classes_):
    cls_name = le.inverse_transform([cls_int])[0]
    print(f"    class {cls_int} ({cls_name:<24}): max_diff={per_class_max[i]:.2e}")

n_proba_mismatch_1pct = int((proba_diff > 0.01).any(axis=1).sum())
n_proba_mismatch_01pct = int((proba_diff > 0.001).any(axis=1).sum())
print(f"\n  Samples with any prob diff > 1%  : {n_proba_mismatch_1pct}/{n_total}")
print(f"  Samples with any prob diff > 0.1%: {n_proba_mismatch_01pct}/{n_total}")

# Class-level breakdown of predictions
print(f"\nPer-class prediction breakdown:")
print(f"  {'Class':<28} {'True':>6} {'SK_pred':>8} {'ONNX_pred':>10} {'Match':>6}")
y_true_str = df_test['RiskLevel'].values
for cls in CANONICAL:
    true_n = int((y_true_str == cls).sum())
    sk_n   = int((y_sk_str   == cls).sum())
    onnx_n = int((y_onnx_str == cls).sum())
    match  = sk_n == onnx_n
    print(f"  {cls:<28} {true_n:>6} {sk_n:>8} {onnx_n:>10} {'YES' if match else 'NO':>6}")

# ONNX output class mapping summary
print(f"\nONNX output class mapping (confirmed):")
print(f"  ONNX outputs integer labels using LabelEncoder alphabetical encoding:")
for cls_int in sorted(clf_audit.classes_):
    cls_name = le.inverse_transform([cls_int])[0]
    print(f"    ONNX integer {cls_int} -> {cls_name}")
print(f"  VERY_UNHEALTHY is NOT in ONNX output (never trained)")
print(f"  Output tensor dimensions: label=(n,), proba=list of {clf_audit.n_classes_}-key dicts")

# Save results
results = {
    "check_A_models_identical": params_match,
    "check_A_scaler_consistent": bool(mean_diff < 1e-6 and scale_diff < 1e-6),
    "check_B_n_total": n_total,
    "check_B_n_match": n_match,
    "check_B_n_mismatch": n_mismatch,
    "check_B_agreement_pct": round(n_match/n_total*100, 6),
    "check_B_max_proba_diff": round(max_abs_diff, 8),
    "check_B_mean_proba_diff": round(mean_abs_diff, 10),
    "check_B_pass": n_mismatch == 0,
    "onnx_input_name": in_name,
    "onnx_output_names": out_names,
    "onnx_label_encoding": {str(int(c)): le.inverse_transform([c])[0] for c in clf_audit.classes_},
}
out_path = os.path.join(AUDIT, "pre_integration_checks.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")
print(f"\nFINAL STATUS:")
print(f"  Check A (artifact consistency): {'PASS' if params_match else 'FAIL'}")
print(f"  Check B (782-sample equivalence): {'PASS' if n_mismatch == 0 else 'FAIL'}")
print("Done.")
