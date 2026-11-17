# -*- coding: utf-8 -*-
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import numpy as np, json, joblib
import onnxruntime as rt

AUDIT = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit"
clf    = joblib.load(f"{AUDIT}\\audit_model_c.pkl")
scaler = joblib.load(f"{AUDIT}\\audit_scaler.pkl")
le     = joblib.load(f"{AUDIT}\\audit_label_encoder.pkl")
sess   = rt.InferenceSession(f"{AUDIT}\\audit_model_c_opset17.onnx")
in_name = sess.get_inputs()[0].name
label_out = sess.get_outputs()[0].name
proba_out = sess.get_outputs()[1].name

FEAT_KEYS = ["Temperature(\u00b0C)","Humidity(%)","Pressure(hPa)",
             "Gas (kOhms)","Altitude(m)","DewPoint(\u00b0C)","HeatIndex(\u00b0C)"]

with open(f"{AUDIT}\\test_vectors.json", encoding='utf-8') as f:
    tv = json.load(f)

# Find the vector with index=1 (row index from test set, array pos 8)
for v in tv['vectors']:
    if v['index'] == 1:
        raw = np.array([v['raw_features'].get(k,0.0) for k in FEAT_KEYS])
        scaled_f64 = scaler.transform(raw.reshape(1,-1))[0]
        scaled_f32 = scaled_f64.astype(np.float32)
        
        sk_proba = clf.predict_proba(scaled_f64.reshape(1,-1))[0]
        ort_res  = sess.run([label_out,proba_out], {in_name: scaled_f32.reshape(1,-1)})
        ort_proba = ort_res[1][0]
        
        print(f"Vector index={v['index']} true={v['true_class']} pred={v['predicted_class']}")
        print(f"JSON proba (probabilities_by_encoded_int): {dict(v['probabilities_by_encoded_int'])}")
        print(f"JSON proba (opset22 backup): {dict(v.get('probabilities_by_encoded_int_opset22',{}))}")
        print()
        print(f"{'Class':<24} {'SK_f64':>10} {'ORT17_f32':>10} {'JSON':>10}")
        for i, cls_int in enumerate(clf.classes_):
            cls = le.inverse_transform([cls_int])[0]
            json_p = v['probabilities_by_encoded_int'].get(cls, 0.0)
            print(f"{cls:<24} {sk_proba[i]:>10.6f} {ort_proba[i]:>10.6f} {json_p:>10.6f}")
        
        print(f"\nAndroid reported GOOD prob: 0.030823")
        print(f"JSON GOOD prob:             {v['probabilities_by_encoded_int'].get('GOOD',0):.6f}")
        print(f"ORT17 GOOD prob:            {ort_proba[0]:.6f}")
        print(f"Are JSON and ORT17 same? {abs(v['probabilities_by_encoded_int'].get('GOOD',0) - ort_proba[0]) < 1e-6}")
        break
