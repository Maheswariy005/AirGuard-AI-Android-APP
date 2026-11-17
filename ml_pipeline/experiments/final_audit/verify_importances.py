# -*- coding: utf-8 -*-
"""Extract feature_importances_ from both PKL models and verify ONNX lineage."""
import sys, io, os, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import joblib

# ── Paths ────────────────────────────────────────────────────
BASE = r"c:\Users\ADMIN\AppData\AirGuardAI"
ORIGINAL_MODEL  = os.path.join(BASE, "ml_pipeline", "airguard_model.pkl")
BEST_MODEL      = os.path.join(BASE, "ml_pipeline", "experiments", "best_model.pkl")
AUDIT_MODEL     = os.path.join(BASE, "ml_pipeline", "experiments", "final_audit", "audit_model_c.pkl")
ONNX_PATH       = os.path.join(BASE, "ml_pipeline", "experiments", "final_audit", "audit_model_c_opset17.onnx")
ONNX_ASSETS     = os.path.join(BASE, "app", "src", "main", "assets", "airguard_model.onnx")

FEATURES = [
    "Temperature(\u00b0C)",
    "Humidity(%)",
    "Pressure(hPa)",
    "Gas (kOhms)",
    "Altitude(m)",
    "DewPoint(\u00b0C)",
    "HeatIndex(\u00b0C)",
]

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ── 1. Original model (from first training, 80/20 split) ────
print("="*72)
print("1. ORIGINAL MODEL (ml_pipeline/airguard_model.pkl)")
print("="*72)
if os.path.exists(ORIGINAL_MODEL):
    clf_orig = joblib.load(ORIGINAL_MODEL)
    imps = clf_orig.feature_importances_
    print(f"   n_estimators: {clf_orig.n_estimators}")
    print(f"   class_weight: {clf_orig.class_weight}")
    print(f"   random_state: {clf_orig.random_state}")
    print(f"   n_classes_: {clf_orig.n_classes_}")
    print(f"   Feature importances (order: {FEATURES}):")
    for f, v in zip(FEATURES, imps):
        print(f"     {f:<26}: {v:.15f}  ({v*100:.4f}%)")
    print(f"   Sum: {sum(imps):.15f}")
else:
    print("   FILE NOT FOUND")

# ── 2. Best model (from experiment, 2026-06-01 split) ───────
print(f"\n{'='*72}")
print("2. BEST MODEL (experiments/best_model.pkl)")
print("="*72)
clf_best = joblib.load(BEST_MODEL)
imps_best = clf_best.feature_importances_
print(f"   n_estimators: {clf_best.n_estimators}")
print(f"   class_weight: {clf_best.class_weight}")
print(f"   random_state: {clf_best.random_state}")
print(f"   n_classes_: {clf_best.n_classes_}")
print(f"   Feature importances (order: {FEATURES}):")
for f, v in zip(FEATURES, imps_best):
    print(f"     {f:<26}: {v:.15f}  ({v*100:.4f}%)")
print(f"   Sum: {sum(imps_best):.15f}")

# ── 3. Audit model (from final audit, same split) ──────────
print(f"\n{'='*72}")
print("3. AUDIT MODEL (experiments/final_audit/audit_model_c.pkl)")
print("="*72)
clf_audit = joblib.load(AUDIT_MODEL)
imps_audit = clf_audit.feature_importances_
print(f"   n_estimators: {clf_audit.n_estimators}")
print(f"   class_weight: {clf_audit.class_weight}")
print(f"   random_state: {clf_audit.random_state}")
print(f"   n_classes_: {clf_audit.n_classes_}")
print(f"   Feature importances (order: {FEATURES}):")
for f, v in zip(FEATURES, imps_audit):
    print(f"     {f:<26}: {v:.15f}  ({v*100:.4f}%)")
print(f"   Sum: {sum(imps_audit):.15f}")

# ── 4. MD5 comparison ───────────────────────────────────────
print(f"\n{'='*72}")
print("4. MD5 COMPARISON (model identity check)")
print("="*72)
for label, path in [
    ("best_model.pkl", BEST_MODEL),
    ("audit_model_c.pkl", AUDIT_MODEL),
    ("ONNX (pipeline)", ONNX_PATH),
    ("ONNX (assets)", ONNX_ASSETS),
]:
    if os.path.exists(path):
        print(f"   {label:<25}: {file_md5(path)[:16]}...  {os.path.getsize(path)/1024/1024:.2f} MB")
    else:
        print(f"   {label:<25}: NOT FOUND")

# ── 5. Best vs audit identity ───────────────────────────────
print(f"\n{'='*72}")
print("5. BEST vs AUDIT MODEL IDENTITY")
print("="*72)
same = np.allclose(imps_best, imps_audit, atol=1e-15)
print(f"   Feature importances identical: {same}")
print(f"   Max diff: {np.abs(imps_best - imps_audit).max():.2e}")

# ── 6. Original vs best/audit comparison ─────────────────────
print(f"\n{'='*72}")
print("6. ORIGINAL vs BEST/AUDIT (different splits → different models)")
print("="*72)
if os.path.exists(ORIGINAL_MODEL):
    print(f"   Original Gas importance: {clf_orig.feature_importances_[3]:.6f}")
    print(f"   Best/Audit Gas importance: {clf_best.feature_importances_[3]:.6f}")
    print(f"   Original Humidity importance: {clf_orig.feature_importances_[1]:.6f}")
    print(f"   Best/Audit Humidity importance: {clf_best.feature_importances_[1]:.6f}")
    diff_orig_best = np.abs(clf_orig.feature_importances_ - clf_best.feature_importances_)
    print(f"   Max diff (original vs best): {diff_orig_best.max():.6f}")
    print(f"   These differ because they are DIFFERENT MODELS from DIFFERENT SPLITS:")
    print(f"     Original: 80/20 row-index split (cutoff ~2026-05-01, mid-day)")
    print(f"     Best/Audit: 2026-06-01 calendar-day split (includes HAZARDOUS in training)")
