# -*- coding: utf-8 -*-
"""
Re-export Model C to ONNX with:
  - ZipMap=False  (plain float tensor output, no OnnxMap)
  - target_opset=17 (stable, fully supported by onnxruntime-android 1.20.0)

ONNX Runtime for Android 1.20.0 guarantees support for opset <= 21.
We use 17 for maximum compatibility with all ORT 1.x Android builds.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
import datetime
import onnxruntime as rt

MODEL_PATH   = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c.pkl"
SCALER_PATH  = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_scaler.pkl"
LE_PATH      = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_label_encoder.pkl"
OUT_ONNX     = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c_opset17.onnx"
ASSETS_DEST  = r"c:\Users\ADMIN\AppData\AirGuardAI\app\src\main\assets\airguard_model.onnx"
CSV_PATH     = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"

FEATURES = ["Temperature(\u00b0C)","Humidity(%)","Pressure(hPa)","Gas (kOhms)","Altitude(m)","DewPoint(\u00b0C)","HeatIndex(\u00b0C)"]
CUTOFF = datetime.date(2026, 6, 1)

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LE_PATH)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 7]))]

onnx_model = convert_sklearn(
    clf,
    initial_types=initial_type,
    target_opset=17,                     # stable opset, supported by all ORT 1.x Android
    options={type(clf): {'zipmap': False}}
)

with open(OUT_ONNX, 'wb') as f:
    f.write(onnx_model.SerializeToString())

size_mb = os.path.getsize(OUT_ONNX) / 1024 / 1024
print(f"ONNX (opset=17, ZipMap=False) saved: {OUT_ONNX}")
print(f"Size: {size_mb:.2f} MB")

# Inspect opset
import onnx
m = onnx.load(OUT_ONNX)
print(f"IR version: {m.ir_version}")
print(f"Opset: {m.opset_import[0].version}")

# Inspect outputs
sess = rt.InferenceSession(OUT_ONNX)
print(f"Outputs:")
for o in sess.get_outputs():
    print(f"  {o.name}: {o.type}")

# Verify 782-sample accuracy
def aqi_to_risk(aqi):
    if aqi<=20: return "GOOD"
    elif aqi<=40: return "MODERATE"
    elif aqi<=60: return "UNHEALTHY_SENSITIVE"
    elif aqi<=80: return "UNHEALTHY"
    elif aqi<=90: return "VERY_UNHEALTHY"
    else: return "HAZARDOUS"

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig').replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
for col in feat_cols+['AQI']: df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates().dropna(subset=feat_cols+['AQI'])
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['_datetime']).sort_values('_datetime').reset_index(drop=True)
df['_date'] = df['_datetime'].dt.date
df_test = df[df['_date'] >= CUTOFF].copy().reset_index(drop=True)
X_test = df_test[FEATURES].values
X_test_sc = scaler.transform(X_test).astype(np.float32)

in_name    = sess.get_inputs()[0].name
label_out  = sess.get_outputs()[0].name
proba_out  = sess.get_outputs()[1].name

results    = sess.run([label_out, proba_out], {in_name: X_test_sc})
onnx_labels = results[0]  # shape (782,)
onnx_probas = results[1]  # shape (782, 5)

sk_labels = clf.predict(X_test_sc)
sk_probas  = clf.predict_proba(X_test_sc).astype(np.float32)

label_agree = int((onnx_labels == sk_labels).sum())
proba_diff  = np.abs(onnx_probas - sk_probas)
max_diff    = float(proba_diff.max())

print(f"\n782-sample verification (opset=17, ZipMap=False):")
print(f"  Label agreement: {label_agree}/782 ({label_agree/782*100:.2f}%)")
print(f"  Max proba diff : {max_diff:.2e}")
print(f"  Proba shape    : {onnx_probas.shape}")
print(f"  Label dtype    : {onnx_labels.dtype}")
print(f"  ONNX class order: {[le.inverse_transform([i])[0] for i in range(clf.n_classes_)]}")

if label_agree == 782 and max_diff < 1e-5:
    import shutil
    shutil.copy2(OUT_ONNX, ASSETS_DEST)
    asset_sz = os.path.getsize(ASSETS_DEST)/1024/1024
    print(f"\nDeployed to: {ASSETS_DEST} ({asset_sz:.2f} MB)")
    print("EXPORT_OK")
else:
    print(f"\nFAILED: agree={label_agree}, max_diff={max_diff}")
