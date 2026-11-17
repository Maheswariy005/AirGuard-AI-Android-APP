# -*- coding: utf-8 -*-
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import numpy as np, joblib

scaler = joblib.load(r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_scaler.pkl")
clf    = joblib.load(r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_model_c.pkl")
le     = joblib.load(r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\audit_label_encoder.pkl")

FEATURES = ["Temperature(°C)","Humidity(%)","Pressure(hPa)","Gas (kOhms)","Altitude(m)","DewPoint(°C)","HeatIndex(°C)"]

samples = [
    ("GOOD",                 [33.2, 71.5, 1006.2, 30.9,  58.8, 27.5, 42.5]),
    ("MODERATE",             [33.1, 70.3, 1006.2, 58.5,  58.5, 27.2, 41.9]),
    ("UNHEALTHY_SENSITIVE",  [33.4, 69.4, 1006.2, 114.4, 58.5, 27.2, 42.0]),
    ("UNHEALTHY",            [33.9, 67.7, 1006.2, 204.5, 58.6, 27.5, 42.5]),
    ("HAZARDOUS input",      [31.5, 52.8, 996.0,  478.1, 144.6,22.0, 33.8]),
]

for label, raw in samples:
    raw_arr = np.array(raw).reshape(1,-1)
    scaled  = scaler.transform(raw_arr)[0]
    pred_int= clf.predict(scaled.reshape(1,-1))[0]
    pred_str= le.inverse_transform([pred_int])[0]
    print(f"\n{label}:")
    print(f"  raw    : {raw}")
    print(f"  scaled : {[round(float(v),4) for v in scaled]}")
    print(f"  pred   : {pred_str}")
