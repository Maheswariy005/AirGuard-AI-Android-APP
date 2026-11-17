# -*- coding: utf-8 -*-
"""
Deep inspection of the verified Model C artifact.
Investigates sklearn RF prediction semantics, leaf structure,
class weight effects, and ONNX conversion feasibility.
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH  = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c.pkl"
SCALER_PATH = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_scaler.pkl"
LE_PATH     = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_label_encoder.pkl"
OUT_DIR     = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit"

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LE_PATH)

FEATURES = [
    "Temperature(°C)","Humidity(%)","Pressure(hPa)",
    "Gas (kOhms)","Altitude(m)","DewPoint(°C)","HeatIndex(°C)",
]

SEP = "="*72

# ══════════════════════════════════════════════════════════════
# 1. Basic model structure
# ══════════════════════════════════════════════════════════════
print(SEP)
print("1. MODEL STRUCTURE")
print(SEP)
print(f"n_estimators    : {clf.n_estimators}")
print(f"n_features_in_  : {clf.n_features_in_}")
print(f"n_classes_      : {clf.n_classes_}")
print(f"classes_        : {clf.classes_}  (LabelEncoder integer indices)")
print(f"class_weight    : {clf.class_weight}")
print(f"random_state    : {clf.random_state}")
print(f"max_features    : {clf.max_features}")
print()
print("LabelEncoder mapping (alphabetical):")
for i, c in enumerate(le.classes_):
    print(f"  {i} -> {c}")
print()
print(f"Features (order matters for inference):")
for i, f in enumerate(FEATURES):
    print(f"  [{i}] {f}")

# ── Tree depths and node counts ──────────────────────────────
depths = []
n_nodes = []
n_leaves = []
for est in clf.estimators_:
    t = est.tree_
    depths.append(t.max_depth)
    n_nodes.append(t.node_count)
    n_leaves.append(int((t.children_left == -1).sum()))

print(f"\nTree depths (across {len(depths)} trees):")
print(f"  Min: {min(depths)}  Max: {max(depths)}  Mean: {np.mean(depths):.1f}  Median: {np.median(depths):.1f}")
print(f"Node counts:")
print(f"  Min: {min(n_nodes)}  Max: {max(n_nodes)}  Mean: {np.mean(n_nodes):.1f}  Total: {sum(n_nodes)}")
print(f"Leaf counts:")
print(f"  Min: {min(n_leaves)}  Max: {max(n_leaves)}  Mean: {np.mean(n_leaves):.1f}  Total: {sum(n_leaves)}")

# ══════════════════════════════════════════════════════════════
# 2. Leaf value structure — class_weight='balanced' effect
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("2. SKLEARN LEAF VALUE STRUCTURE AND PREDICTION SEMANTICS")
print(SEP)

# Inspect tree 0 internal structure
t0 = clf.estimators_[0].tree_
print(f"Tree 0 node_count: {t0.node_count}")
print(f"Tree 0 value shape: {t0.value.shape}  # (n_nodes, n_outputs, n_classes)")
print(f"Tree 0 n_classes: {t0.n_classes}")
print()

# Find first few leaf nodes and show their value arrays
leaf_indices = np.where(t0.children_left == -1)[0]
print(f"Tree 0 has {len(leaf_indices)} leaf nodes. Showing first 8:")
for li in leaf_indices[:8]:
    vals = t0.value[li, 0, :]  # shape (n_classes,)
    total = vals.sum()
    normed = vals / total if total > 0 else vals
    pred_class = int(np.argmax(vals))
    print(f"  Leaf node {li:4d}: raw_values={np.round(vals,3)}  sum={total:.3f}  "
          f"normalized={np.round(normed,3)}  argmax_class={pred_class}({le.classes_[pred_class]})")

print()
print("KEY FINDING: With class_weight='balanced', leaf values are WEIGHTED")
print("class counts (not raw counts). They are NOT integers.")
print("sklearn RF predict_proba() normalizes each tree's leaf and averages across trees.")
print("predict() calls predict_proba() then takes argmax.")
print()

# Demonstrate with a sample prediction
import pandas as pd
CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
import datetime
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df = df.replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
for col in feat_cols + ['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates().dropna(subset=feat_cols+['AQI'])
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['_datetime']).sort_values('_datetime').reset_index(drop=True)
df['_date'] = df['_datetime'].dt.date

CUTOFF = datetime.date(2026, 6, 1)
df_test = df[df['_date'] >= CUTOFF].copy().reset_index(drop=True)
X_test = df_test[FEATURES].values
X_test_sc = scaler.transform(X_test)

# Detailed trace of prediction for sample #0 (a HAZARDOUS sample)
def aqi_to_risk(aqi):
    if aqi<=20: return "GOOD"
    elif aqi<=40: return "MODERATE"
    elif aqi<=60: return "UNHEALTHY_SENSITIVE"
    elif aqi<=80: return "UNHEALTHY"
    elif aqi<=90: return "VERY_UNHEALTHY"
    else: return "HAZARDOUS"
df_test['RiskLevel'] = df_test['AQI'].apply(aqi_to_risk)
haz_idx_df = df_test[df_test['RiskLevel']=='HAZARDOUS'].index[0]
x_sample = X_test_sc[haz_idx_df:haz_idx_df+1]
x_raw    = X_test[haz_idx_df:haz_idx_df+1]

print(f"Sample index (a HAZARDOUS row): {haz_idx_df}")
print(f"  Raw features: {dict(zip(FEATURES, x_raw[0].round(3)))}")
print(f"  Scaled features: {x_sample[0].round(4)}")

proba = clf.predict_proba(x_sample)[0]
pred_int = clf.predict(x_sample)[0]
pred_str = le.inverse_transform([pred_int])[0]
print(f"\n  predict_proba()  -> {dict(zip(le.classes_, proba.round(4)))}")
print(f"  predict()        -> {pred_int} ({pred_str})")
print(f"  argmax(proba)    -> {np.argmax(proba)} ({le.classes_[np.argmax(proba)]})")

# Manual aggregation: collect probabilities from each tree
tree_probas = np.array([est.predict_proba(x_sample)[0] for est in clf.estimators_])
mean_proba = tree_probas.mean(axis=0)
print(f"\n  Manual mean of {len(clf.estimators_)} tree predict_proba(): {mean_proba.round(4)}")
print(f"  Matches clf.predict_proba(): {np.allclose(mean_proba, proba, atol=1e-10)}")
print()
print("CONFIRMED: sklearn RF predict_proba = mean of per-tree predict_proba.")
print("Each tree's predict_proba = leaf_value_normalized / leaf_value_sum.")
print("Final predict = argmax of mean probabilities across all trees.")

# Show the difference from simple majority voting
tree_preds = np.array([est.predict(x_sample)[0] for est in clf.estimators_])
vote_counts = np.bincount(tree_preds, minlength=clf.n_classes_)
majority_pred = np.argmax(vote_counts)
print(f"\n  Simple majority vote counts: {vote_counts}")
print(f"  Majority vote prediction   : {majority_pred} ({le.classes_[majority_pred]})")
print(f"  Probability argmax pred    : {pred_int} ({pred_str})")
agree = majority_pred == pred_int
print(f"  Agree on this sample: {agree}")

# Check agreement across all test samples
y_proba_pred = clf.predict(X_test_sc)
y_majority_pred = np.array([
    np.argmax(np.bincount(
        np.array([est.predict(X_test_sc[i:i+1])[0] for est in clf.estimators_]),
        minlength=clf.n_classes_
    ))
    for i in range(min(100, len(X_test_sc)))  # check first 100 for speed
])
agreement = (y_proba_pred[:100] == y_majority_pred).mean()
print(f"\n  Agreement between proba-argmax and majority-vote on 100 test samples: {agreement*100:.1f}%")
n_disagree = (y_proba_pred[:100] != y_majority_pred).sum()
print(f"  Disagreements: {n_disagree} out of 100")

# ══════════════════════════════════════════════════════════════
# 3. ONNX conversion feasibility
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("3. ONNX CONVERSION FEASIBILITY")
print(SEP)
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print(f"skl2onnx version: {skl2onnx.__version__}")
    
    n_features = len(FEATURES)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    
    onnx_path = os.path.join(OUT_DIR, "audit_model_c.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    onnx_size_mb = os.path.getsize(onnx_path) / (1024*1024)
    print(f"ONNX export: SUCCESS")
    print(f"ONNX file size: {onnx_size_mb:.2f} MB")
    
    # Verify ONNX predictions match sklearn
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        proba_name = sess.get_outputs()[1].name
        
        X_sample = X_test_sc[:20].astype(np.float32)
        onnx_labels = sess.run([label_name], {input_name: X_sample})[0]
        sklearn_labels = clf.predict(X_test_sc[:20])
        
        agreement = (onnx_labels == sklearn_labels).mean()
        print(f"ONNX vs sklearn agreement on 20 samples: {agreement*100:.1f}%")
        print(f"ONNX available: YES. Exact prediction match: {agreement == 1.0}")
        ONNX_AVAILABLE = True
    except ImportError:
        print("onnxruntime not installed. ONNX file created but cannot verify predictions.")
        ONNX_AVAILABLE = "file_only"
    
except ImportError:
    print("skl2onnx not installed.")
    ONNX_AVAILABLE = False
    onnx_size_mb = None
    print("Attempting install hint: pip install skl2onnx onnxruntime")

# ══════════════════════════════════════════════════════════════
# 4. Model size comparison
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. MODEL SIZE AND DEPLOYMENT ESTIMATES")
print(SEP)
pkl_size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
print(f"PKL model size     : {pkl_size_mb:.2f} MB")
if onnx_size_mb:
    print(f"ONNX model size    : {onnx_size_mb:.2f} MB")

total_nodes = sum(n_nodes)
total_leaves = sum(n_leaves)
# Kotlin code size estimate: each leaf ~ 30 chars, each split ~ 80 chars
internal_nodes = total_nodes - total_leaves
kt_size_estimate_kb = (total_leaves * 30 + internal_nodes * 80) / 1024
print(f"\nTotal tree nodes   : {total_nodes}")
print(f"  Split nodes      : {internal_nodes}")
print(f"  Leaf nodes       : {total_leaves}")
print(f"Kotlin code estimate: ~{kt_size_estimate_kb:.0f} KB of source code")
print(f"  (This is data, not logic — it's very large Kotlin files)")

# ══════════════════════════════════════════════════════════════
# 5. Test vector export
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. TEST VECTOR EXPORT (for Kotlin equivalence testing)")
print(SEP)

# Export 50 test samples with raw features, scaled features, and expected predictions
n_export = 50
indices_per_class = {}
for cls in le.classes_:
    cls_str = le.inverse_transform([le.transform([cls])[0]])[0]
    df_cls = df_test[df_test['RiskLevel']==cls_str]
    if len(df_cls) > 0:
        indices_per_class[cls_str] = df_cls.index[:min(10, len(df_cls))].tolist()

export_rows = []
for cls_str, idxs in indices_per_class.items():
    for idx in idxs:
        raw = X_test[idx]
        scaled = X_test_sc[idx]
        pred_int = clf.predict(scaled.reshape(1,-1))[0]
        pred_str = le.inverse_transform([pred_int])[0]
        proba = clf.predict_proba(scaled.reshape(1,-1))[0]
        export_rows.append({
            "index": int(idx),
            "true_class": df_test.loc[idx, 'RiskLevel'],
            "predicted_class": pred_str,
            "raw_features": {f: round(float(v),4) for f,v in zip(FEATURES,raw)},
            "scaled_features": {f: round(float(v),6) for f,v in zip(FEATURES,scaled)},
            "probabilities": {le.classes_[i]: round(float(p),6) for i,p in enumerate(proba)},
            "predicted_int": int(pred_int),
        })

tv_path = os.path.join(OUT_DIR, "test_vectors.json")
with open(tv_path, "w", encoding="utf-8") as f:
    json.dump({
        "description": "Test vectors for Kotlin equivalence testing",
        "canonical_classes": ["GOOD","MODERATE","UNHEALTHY_SENSITIVE","UNHEALTHY","VERY_UNHEALTHY","HAZARDOUS"],
        "label_encoder_alphabetical": list(le.classes_),
        "features_in_order": FEATURES,
        "scaler_means": {f: round(float(m),8) for f,m in zip(FEATURES,scaler.mean_)},
        "scaler_scales": {f: round(float(s),8) for f,s in zip(FEATURES,scaler.scale_)},
        "n_vectors": len(export_rows),
        "vectors": export_rows
    }, f, indent=2)
print(f"Exported {len(export_rows)} test vectors to: {tv_path}")

# Class coverage of test vectors
exported_classes = set(r['true_class'] for r in export_rows)
print(f"Classes covered in test vectors: {exported_classes}")

print("\nDone.")
