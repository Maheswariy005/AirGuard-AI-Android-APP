# -*- coding: utf-8 -*-
"""Part 2: majority-vote comparison, ONNX, model sizes, test vector export."""
import sys, io, json, os, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib

MODEL_PATH  = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c.pkl"
SCALER_PATH = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_scaler.pkl"
LE_PATH     = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_label_encoder.pkl"
OUT_DIR     = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit"
CSV_PATH    = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
FEATURES = [
    "Temperature(\u00b0C)","Humidity(%)","Pressure(hPa)",
    "Gas (kOhms)","Altitude(m)","DewPoint(\u00b0C)","HeatIndex(\u00b0C)",
]
CANONICAL = ["GOOD","MODERATE","UNHEALTHY_SENSITIVE","UNHEALTHY","VERY_UNHEALTHY","HAZARDOUS"]

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LE_PATH)

# Rebuild test set
def aqi_to_risk(aqi):
    if aqi<=20: return "GOOD"
    elif aqi<=40: return "MODERATE"
    elif aqi<=60: return "UNHEALTHY_SENSITIVE"
    elif aqi<=80: return "UNHEALTHY"
    elif aqi<=90: return "VERY_UNHEALTHY"
    else: return "HAZARDOUS"

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig').replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
for col in feat_cols+['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates().dropna(subset=feat_cols+['AQI'])
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['_datetime']).sort_values('_datetime').reset_index(drop=True)
df['_date'] = df['_datetime'].dt.date
df['RiskLevel'] = df['AQI'].apply(aqi_to_risk)
CUTOFF = datetime.date(2026, 6, 1)
df_test = df[df['_date'] >= CUTOFF].copy().reset_index(drop=True)
X_test = df_test[FEATURES].values
X_test_sc = scaler.transform(X_test)

SEP = "="*72

# ══════════════════════════════════════════════════════════════
# Majority vote vs proba-argmax comparison
# ══════════════════════════════════════════════════════════════
print(SEP)
print("MAJORITY VOTE vs PROBABILITY ARGMAX")
print(SEP)

# proba-argmax (sklearn default)
y_proba_pred = clf.predict(X_test_sc)

# majority vote: per tree predict -> bincount -> argmax
disagree_count = 0
disagree_examples = []
for i in range(len(X_test_sc)):
    tree_preds = np.array([est.predict(X_test_sc[i:i+1])[0] for est in clf.estimators_], dtype=int)
    vote_counts = np.bincount(tree_preds, minlength=clf.n_classes_)
    majority_pred = int(np.argmax(vote_counts))
    proba_pred    = int(y_proba_pred[i])
    if majority_pred != proba_pred:
        disagree_count += 1
        if len(disagree_examples) < 3:
            disagree_examples.append({
                'i': i, 'majority': int(majority_pred), 'proba': int(proba_pred),
                'majority_str': le.inverse_transform([majority_pred])[0],
                'proba_str': le.inverse_transform([proba_pred])[0],
                'vote_counts': vote_counts.tolist()
            })

n_total = len(X_test_sc)
print(f"Total test samples: {n_total}")
print(f"Disagreements (majority vote != proba argmax): {disagree_count} / {n_total} ({disagree_count/n_total*100:.2f}%)")
if disagree_examples:
    print("Example disagreements:")
    for ex in disagree_examples:
        print(f"  sample {ex['i']}: majority={ex['majority_str']}  proba_argmax={ex['proba_str']}  votes={ex['vote_counts']}")
else:
    print("No disagreements on this test set.")

# ══════════════════════════════════════════════════════════════
# ONNX
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ONNX EXPORT")
print(SEP)

try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print(f"skl2onnx: {skl2onnx.__version__}")
    initial_type = [('float_input', FloatTensorType([None, len(FEATURES)]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    onnx_path = os.path.join(OUT_DIR, "audit_model_c.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    onnx_mb = os.path.getsize(onnx_path) / (1024*1024)
    print(f"ONNX export SUCCESS. Size: {onnx_mb:.2f} MB at {onnx_path}")
    
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        X_sample = X_test_sc[:100].astype(np.float32)
        onnx_preds = sess.run([label_name], {input_name: X_sample})[0]
        sk_preds = clf.predict(X_test_sc[:100])
        agree = (onnx_preds == sk_preds).mean()
        print(f"onnxruntime verification on 100 samples: {agree*100:.1f}% agreement")
        print(f"Exact match: {agree == 1.0}")
    except ImportError:
        print("onnxruntime not installed — cannot verify predictions")
    except Exception as e:
        print(f"onnxruntime error: {e}")

except ImportError:
    print("skl2onnx not available.")
    onnx_mb = None
except Exception as e:
    print(f"ONNX export failed: {e}")
    onnx_mb = None

# ══════════════════════════════════════════════════════════════
# Model sizes
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("MODEL SIZE COMPARISON")
print(SEP)
pkl_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
print(f"PKL size: {pkl_mb:.2f} MB")
n_nodes  = sum(e.tree_.node_count for e in clf.estimators_)
n_leaves = sum(int((e.tree_.children_left==-1).sum()) for e in clf.estimators_)
n_splits = n_nodes - n_leaves
# Kotlin estimate: each Split node ~100 chars, each Leaf node ~40 chars
kt_est_kb = (n_splits * 100 + n_leaves * 40) / 1024
print(f"Total nodes : {n_nodes}  (splits: {n_splits}, leaves: {n_leaves})")
print(f"Kotlin source estimate: ~{kt_est_kb:.0f} KB")
print(f"Kotlin APK compiled estimate: ~{kt_est_kb*0.3:.0f}-{kt_est_kb*0.5:.0f} KB (rough)")

# ══════════════════════════════════════════════════════════════
# Test vector export
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TEST VECTOR EXPORT")
print(SEP)

rows = []
for cls in CANONICAL:
    cls_rows = df_test[df_test['RiskLevel']==cls]
    for idx in cls_rows.index[:min(10, len(cls_rows))]:
        raw   = X_test[idx]
        sc    = X_test_sc[idx]
        p_int = int(clf.predict(sc.reshape(1,-1))[0])
        p_str = le.inverse_transform([p_int])[0]
        proba = clf.predict_proba(sc.reshape(1,-1))[0]
        proba_dict = {le.classes_[i]: round(float(v),6) for i,v in enumerate(proba)}
        rows.append({
            "index": int(idx),
            "true_class": cls,
            "predicted_class": p_str,
            "correct": p_str == cls,
            "raw_features": {f: round(float(v),4) for f,v in zip(FEATURES,raw)},
            "scaled_features": {f: round(float(v),6) for f,v in zip(FEATURES,sc)},
            "probabilities_by_encoded_int": proba_dict,
            "predicted_encoded_int": p_int,
        })

tv_path = os.path.join(OUT_DIR, "test_vectors.json")
out_data = {
    "description": "Ground-truth test vectors for Kotlin equivalence testing",
    "model": "RF class_weight=balanced, 200 trees, random_state=42",
    "split_cutoff": "2026-06-01",
    "canonical_class_order": CANONICAL,
    "label_encoder_alphabetical_classes": list(le.classes_),
    "label_encoder_mapping": {str(i): c for i,c in enumerate(le.classes_)},
    "features_in_order": FEATURES,
    "scaler_means":  {f: round(float(m),8) for f,m in zip(FEATURES,scaler.mean_)},
    "scaler_scales": {f: round(float(s),8) for f,s in zip(FEATURES,scaler.scale_)},
    "scaling_formula": "scaled = (raw - mean) / scale",
    "prediction_formula": "predict = argmax(mean(per_tree_predict_proba))",
    "n_vectors": len(rows),
    "vectors": rows,
}
with open(tv_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)
covered = set(r['true_class'] for r in rows)
correct_pct = sum(1 for r in rows if r['correct']) / len(rows) * 100
print(f"Exported {len(rows)} vectors. Classes covered: {covered}")
print(f"Correct predictions in test vectors: {correct_pct:.1f}%")
print(f"Saved to: {tv_path}")
print("\nDone.")
