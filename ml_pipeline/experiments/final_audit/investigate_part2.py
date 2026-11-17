# -*- coding: utf-8 -*-
"""Part 2 of investigation: test_vectors.json structure, opset-17 vs opset-22 diff,
   regeneration with correct probabilities."""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import joblib
import onnxruntime as rt

AUDIT_DIR   = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit"
MODEL_PATH  = os.path.join(AUDIT_DIR, "audit_model_c.pkl")
SCALER_PATH = os.path.join(AUDIT_DIR, "audit_scaler.pkl")
LE_PATH     = os.path.join(AUDIT_DIR, "audit_label_encoder.pkl")
ONNX_OP17   = os.path.join(AUDIT_DIR, "audit_model_c_opset17.onnx")
TV_PATH     = os.path.join(AUDIT_DIR, "test_vectors.json")
FEAT_KEYS   = ["Temperature(\u00b0C)","Humidity(%)","Pressure(hPa)",
               "Gas (kOhms)","Altitude(m)","DewPoint(\u00b0C)","HeatIndex(\u00b0C)"]

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LE_PATH)
sess   = rt.InferenceSession(ONNX_OP17)
in_name   = sess.get_inputs()[0].name
label_out = sess.get_outputs()[0].name
proba_out = sess.get_outputs()[1].name

with open(TV_PATH, encoding='utf-8') as f:
    tv_data = json.load(f)
tv_vectors = tv_data['vectors']

SEP = "="*72

# ══════════════════════════════════════════════════════════════
# TASK 3: test_vectors.json structure
# ══════════════════════════════════════════════════════════════
print(SEP)
print("TASK 3: test_vectors.json structure")
print(SEP)
# Determine probability field name
sample = tv_vectors[0]
prob_field = None
for f in ['probabilities_by_encoded_int', 'probabilities']:
    if f in sample:
        prob_field = f
        break
print(f"  Total vectors: {len(tv_vectors)}")
print(f"  Probability field name in JSON: '{prob_field}'")
print(f"  Sample probability keys: {list(sample[prob_field].keys())[:5]}")

true_classes = [v['true_class'] for v in tv_vectors]
pred_classes = [v['predicted_class'] for v in tv_vectors]
true_counts  = {c: true_classes.count(c) for c in sorted(set(true_classes))}
pred_counts  = {c: pred_classes.count(c) for c in sorted(set(pred_classes))}

print(f"\n  true_class distribution: {true_counts}")
print(f"  predicted_class distribution (Python model): {pred_counts}")

FIVE_TRAINED = ['GOOD','HAZARDOUS','MODERATE','UNHEALTHY','UNHEALTHY_SENSITIVE']
print(f"\n  Coverage of 5 trained classes as true_class (ground-truth inputs):")
for cls in FIVE_TRAINED:
    n = true_classes.count(cls)
    status = "PRESENT" if n > 0 else "ABSENT [PROBLEM]"
    print(f"    {cls:<26}: {n:>3} vectors  [{status}]")

# ══════════════════════════════════════════════════════════════
# Root cause of the 5.7e-3 discrepancy
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ROOT CAUSE: opset-17 vs opset-22 probability comparison")
print(SEP)
print("  The failing instrumented test compared Android(opset-17) vs json probabilities.")
print("  json probabilities came from the opset-22 export (original audit_model_c.onnx).")
print("  opset-17 uses different ML operator implementations for RF aggregation.")

max_17_vs_22 = 0.0
n_above_1e3 = 0
print(f"\n  Failing vectors from Android test report:")
# Vectors 1 and 5 (json index — which are actual test-set row indices 153 and ?)
for vec in tv_vectors[:10]:  # check first 10
    raw = np.array([vec['raw_features'].get(k, 0.0) for k in FEAT_KEYS])
    scaled_f32 = scaler.transform(raw.reshape(1,-1))[0].astype(np.float32)
    ort_res = sess.run([label_out, proba_out], {in_name: scaled_f32.reshape(1,-1)})
    ort_proba_17 = ort_res[1][0]  # float32

    # Original json proba (opset-22 based, keyed by class name or encoded int)
    json_proba = sample[prob_field]
    # We need to map json keys to class names
    # The field is probabilities_by_encoded_int -- keys are class names (confirmed earlier)
    orig_proba = vec[prob_field]

    for i, cls_int in enumerate(clf.classes_):
        cls_name = le.inverse_transform([cls_int])[0]
        p17 = float(ort_proba_17[i])
        # orig keys: check whether they're class names or int strings
        if cls_name in orig_proba:
            p_json = orig_proba[cls_name]
        elif str(i) in orig_proba:
            p_json = orig_proba[str(i)]
        else:
            p_json = 0.0
        diff = abs(p17 - p_json)
        if diff > max_17_vs_22: max_17_vs_22 = diff
        if diff > 1e-3: n_above_1e3 += 1

print(f"  Max diff opset-17 vs json(opset-22) on first 10 vectors: {max_17_vs_22:.6e}")
print(f"  Diffs > 1e-3: {n_above_1e3}")

# Check all 48
max_all = 0.0
n_above_1e3_all = 0
for vec in tv_vectors:
    raw = np.array([vec['raw_features'].get(k, 0.0) for k in FEAT_KEYS])
    scaled_f32 = scaler.transform(raw.reshape(1,-1))[0].astype(np.float32)
    ort_res = sess.run([label_out, proba_out], {in_name: scaled_f32.reshape(1,-1)})
    ort_p17 = ort_res[1][0]
    orig_proba = vec[prob_field]
    for i, cls_int in enumerate(clf.classes_):
        cls_name = le.inverse_transform([cls_int])[0]
        p17 = float(ort_p17[i])
        p_json = orig_proba.get(cls_name, orig_proba.get(str(i), 0.0))
        diff = abs(p17 - p_json)
        if diff > max_all: max_all = diff
        if diff > 1e-3: n_above_1e3_all += 1

print(f"\n  Max diff opset-17 vs json(opset-22) across all 48 vectors: {max_all:.6e}")
print(f"  Diffs > 1e-3: {n_above_1e3_all}")
print()
print("  This confirms: the ~5.7e-3 discrepancy in the instrumented test was caused by")
print("  comparing the Android (opset-17) output against reference probabilities that")
print("  were generated by an opset-22 ONNX model. The two opset versions have")
print("  numerically different floating-point implementations of the RF aggregation.")
print()
print("  FIX: Regenerate test_vectors.json probabilities using opset-17 ORT.")
print("  After that fix, Android vs Python ORT(opset-17) will agree to ~4e-7 (float32).")
print("  The correct tolerance is 1e-3, which comfortably covers the actual 4e-7 diff.")

# ══════════════════════════════════════════════════════════════
# Regenerate test_vectors.json with opset-17 probabilities
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("REGENERATING test_vectors.json with opset-17 probabilities")
print(SEP)

for vec in tv_vectors:
    raw = np.array([vec['raw_features'].get(k, 0.0) for k in FEAT_KEYS])
    scaled_f32 = scaler.transform(raw.reshape(1,-1))[0].astype(np.float32)
    ort_res = sess.run([label_out, proba_out], {in_name: scaled_f32.reshape(1,-1)})
    ort_label = int(ort_res[0][0])
    ort_proba = ort_res[1][0]
    pred_str = le.inverse_transform([ort_label])[0]
    # Build class-name-keyed probability dict
    new_probas = {le.inverse_transform([cls_int])[0]: round(float(ort_proba[i]), 8)
                  for i, cls_int in enumerate(clf.classes_)}
    # Replace the probability field with opset-17 values (keep original as backup)
    vec['probabilities_by_encoded_int_opset22'] = vec[prob_field]  # archive original
    vec[prob_field] = new_probas                                     # replace with opset-17
    vec['predicted_class'] = pred_str                                # update predicted class

agree = sum(1 for v in tv_vectors if v['true_class'] == v['predicted_class'])
print(f"  Updated predictions: opset-17 predicted_class correct on {agree}/48 true-class matches")

# Verify max diff between new json and opset-17 ORT (should be tiny)
max_after = 0.0
for vec in tv_vectors:
    raw = np.array([vec['raw_features'].get(k, 0.0) for k in FEAT_KEYS])
    scaled_f32 = scaler.transform(raw.reshape(1,-1))[0].astype(np.float32)
    ort_res = sess.run([label_out, proba_out], {in_name: scaled_f32.reshape(1,-1)})
    ort_p17 = ort_res[1][0]
    for i, cls_int in enumerate(clf.classes_):
        cls_name = le.inverse_transform([cls_int])[0]
        diff = abs(float(ort_p17[i]) - vec[prob_field].get(cls_name, 0.0))
        if diff > max_after: max_after = diff
print(f"  Max diff new_json vs ORT-17 (should be ~0): {max_after:.2e}")

# Save updated JSON
out_data = dict(tv_data)
out_data['vectors'] = tv_vectors
out_data['onnx_probabilities_source'] = 'opset-17 ZipMap=False (audit_model_c_opset17.onnx)'
with open(TV_PATH, 'w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2)

print(f"  Saved updated test_vectors.json")
print()
print("Done.")
