# -*- coding: utf-8 -*-
"""
Re-export the verified Model C to ONNX with ZipMap disabled.
With ZipMap=False the probability output is a plain float32 tensor
of shape [n_samples, n_classes] instead of a sequence of int->float maps.
This is easier to consume on Android and avoids the OnnxMap cast issue.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import joblib
import onnxruntime as rt

MODEL_PATH   = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c.pkl"
SCALER_PATH  = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_scaler.pkl"
LE_PATH      = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_label_encoder.pkl"
OUT_ONNX     = r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c_nozipmap.onnx"
ASSETS_DEST  = r"c:\Users\ADMIN\AppData\AirGuardAI\app\src\main\assets\airguard_model.onnx"

clf    = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LE_PATH)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 7]))]

# ZipMap=False makes output_probability a float32 tensor [n, n_classes]
# instead of a sequence of {int -> float} maps
onnx_model = convert_sklearn(
    clf,
    initial_types=initial_type,
    options={type(clf): {'zipmap': False}}
)

with open(OUT_ONNX, 'wb') as f:
    f.write(onnx_model.SerializeToString())

size_mb = os.path.getsize(OUT_ONNX) / 1024 / 1024
print(f"ONNX (ZipMap=False) saved: {OUT_ONNX}")
print(f"Size: {size_mb:.2f} MB")

# Verify: check outputs
sess = rt.InferenceSession(OUT_ONNX)
print(f"\nOutputs:")
for o in sess.get_outputs():
    print(f"  {o.name}: {o.type}")

# Verify predictions still match sklearn
import pandas as pd, datetime
CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
FEATURES = ["Temperature(\u00b0C)","Humidity(%)","Pressure(hPa)","Gas (kOhms)","Altitude(m)","DewPoint(\u00b0C)","HeatIndex(\u00b0C)"]
CUTOFF = datetime.date(2026, 6, 1)

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
df['RiskLevel'] = df['AQI'].apply(aqi_to_risk)
df_test = df[df['_date'] >= CUTOFF].copy().reset_index(drop=True)
X_test = df_test[FEATURES].values
X_test_sc = scaler.transform(X_test).astype(np.float32)

in_name = sess.get_inputs()[0].name
label_out = sess.get_outputs()[0].name
proba_out = sess.get_outputs()[1].name

results = sess.run([label_out, proba_out], {in_name: X_test_sc})
onnx_labels = results[0]  # shape (782,) int64
onnx_probas = results[1]  # shape (782, 5) float32 — with ZipMap=False

sk_labels = clf.predict(X_test_sc)
sk_probas = clf.predict_proba(X_test_sc).astype(np.float32)

label_agree = (onnx_labels == sk_labels).sum()
proba_diff = np.abs(onnx_probas - sk_probas)
max_diff = float(proba_diff.max())

print(f"\n782-sample verification (ZipMap=False):")
print(f"  Label agreement: {label_agree}/782 ({label_agree/782*100:.2f}%)")
print(f"  Max proba diff : {max_diff:.2e}")
print(f"  Proba shape    : {onnx_probas.shape}")
print(f"  Label dtype    : {onnx_labels.dtype}")
print(f"  Proba dtype    : {onnx_probas.dtype}")
print(f"  ONNX class order (index -> class): {[le.inverse_transform([i])[0] for i in range(clf.n_classes_)]}")

# If verification passes, deploy to assets
if label_agree == 782 and max_diff < 1e-5:
    import shutil
    shutil.copy2(OUT_ONNX, ASSETS_DEST)
    asset_size = os.path.getsize(ASSETS_DEST)/1024/1024
    print(f"\nDeployed to assets: {ASSETS_DEST} ({asset_size:.2f} MB)")
    print("EXPORT_VERIFIED_OK")
else:
    print(f"\nWARNING: verification failed (agree={label_agree}, max_diff={max_diff})")
    print("Not deploying.")
