# AirGuard AI — Android Deployment Architecture Proposal

**Status: PROPOSAL ONLY — awaiting approval before any implementation.**  
**No Android/Kotlin files have been modified.**  
**All findings below come from actual Python inspection of the saved model artifacts and reading the existing Kotlin source.**

---

## 1. Verified Model Artifact Inspection

### Confirmed: `experiments/best_model.pkl` vs `final_audit/audit_model_c.pkl`

Both artifacts are the same experiment (RF balanced, cutoff 2026-06-01, random_state=42) but the `best_model.pkl` was saved from `step3_experiments.py` and `audit_model_c.pkl` from the full reproducibility audit. The audit artifact is authoritative because it was saved from a fully re-verified run with explicit class ordering. Use `audit_model_c.pkl` for deployment.

### Model structure (from artifact)

| Property | Value |
|---|---|
| Algorithm | RandomForestClassifier |
| n_estimators | 200 |
| n_features_in_ | 7 |
| n_classes_ (trained) | **5** (VERY_UNHEALTHY never seen; not in model) |
| class_weight | balanced |
| random_state | 42 |
| max_features | sqrt |

### Classes actually learned (LabelEncoder alphabetical order = model internal order)

| Internal int | Class string |
|---:|---|
| 0 | GOOD |
| 1 | HAZARDOUS |
| 2 | MODERATE |
| 3 | UNHEALTHY |
| 4 | UNHEALTHY_SENSITIVE |
| — | VERY_UNHEALTHY (not in model; was never trained) |

This mapping must be reproduced exactly in any Android implementation. VERY_UNHEALTHY can never be predicted by this model.

### Feature order (must be preserved exactly)

| Index | Feature | Scaler mean | Scaler scale |
|---|---|---:|---:|
| 0 | Temperature(°C) | 32.404256 | 4.205421 |
| 1 | Humidity(%) | 65.030945 | 7.397750 |
| 2 | Pressure(hPa) | 1002.640914 | 5.991025 |
| 3 | Gas (kOhms) | 253.995597 | 89.545656 |
| 4 | Altitude(m) | 88.799590 | 50.336149 |
| 5 | DewPoint(°C) | 25.388473 | 4.758210 |
| 6 | HeatIndex(°C) | 39.360001 | 7.739075 |

### Tree structure statistics

| Metric | Value |
|---|---|
| Tree depth range | 11 – 24 |
| Mean tree depth | 15.7 |
| Total nodes | 52,078 |
| Split nodes | 25,939 |
| Leaf nodes | 26,139 |
| Kotlin source estimate | ~3,554 KB |

---

## 2. Exact sklearn Prediction Semantics

This section is critical. The current Kotlin implementation uses simple majority voting, which is **not** how sklearn RF predicts.

### What sklearn actually does

```
predict(x):
  for each tree t in forest (200 trees):
    traverse tree → reach leaf node
    leaf.value = weighted_class_counts (shape [n_classes])  # floats, NOT ints
    per_tree_proba[t] = leaf.value / leaf.value.sum()       # normalize
  
  mean_proba = mean(per_tree_proba, axis=0)   # shape [5]
  prediction = argmax(mean_proba)
```

The weighted class counts come from `class_weight='balanced'`, which scales each class's count by `n_samples / (n_classes * class_count)`. Leaves contain **float** values, not integer vote counts.

### Why this differs from majority voting

Measured on the 782-row test set: **40 out of 782 samples (5.12%) produce a different result** when using majority voting vs probability argmax.

Example: sample 277 (actual=HAZARDOUS)
- Simple majority vote: HAZARDOUS (71 trees) ties with UNHEALTHY (71 trees) — resolved to HAZARDOUS by tie-breaking
- Probability argmax: UNHEALTHY (the weighted probabilities favor UNHEALTHY because HAZARDOUS leaves have lower weighted probability mass)

This 5.12% disagreement rate means: **if you implement majority voting in Kotlin, your Android model will disagree with the Python model on 1 in 20 predictions**. For safety-critical HAZARDOUS predictions this is unacceptable.

### Exact equivalence requirement

Any Android implementation must reproduce:
`prediction = argmax(mean([normalize(leaf.weighted_counts) for each tree]))`

---

## 3. Existing Kotlin Architecture Analysis

### Current data flow

```
SensorRepository (CSV → SensorReading objects)
    ↓
DashboardViewModel.loadData()
    → classifier.predict(reading)          ← uses AQI as feature index 3
    ↓
PredictionResult (riskLevel, confidence, votes, classProbabilities, features)
    ↓
DashboardFragment (UI)
```

```
AnalysisViewModel.loadAnalysis()
    → classifier.predict(latest)           ← also uses AQI
    → aqiDistribution, riskHistoryData     ← uses r.riskLevel (from SensorReading.aqi)
```

### Current classifier interface

```kotlin
class AirQualityClassifier {
    fun predict(reading: SensorReading): PredictionResult
    fun predictFromFeatures(features: DoubleArray): PredictionResult
    // FEATURE_COUNT = 8, includes AQI at index 3
}
```

### What must change

| Current | New requirement |
|---|---|
| 8 features (AQI at index 3) | 7 features (AQI excluded) |
| 15 hand-authored trees | 200 trained trees |
| `TreeNode.Leaf(classId: Int)` | `TreeNode.Leaf(weights: DoubleArray)` |
| Majority vote aggregation | Mean-probability argmax |
| Feature order: temp,hum,pres,**aqi**,gas,alt,dp,hi | Feature order: temp,hum,pres,gas,alt,dp,hi |
| No StandardScaler | StandardScaler pre-processing required |

### AQI handling after the change

`SensorReading.aqi` and `SensorReading.riskLevel` (computed via `RiskLevel.fromAqi(aqi)`) remain unchanged. AQI is still used to:
- Display the raw AQI value to the user
- Compute the `riskLevel` property for all existing UI that shows the AQI-derived risk label
- Drive `aqiDistribution` in AnalysisViewModel

The **ML model** no longer uses AQI as an input. The app will have two parallel risk assessments:
- `reading.riskLevel` — deterministic AQI-derived label (existing, unchanged)
- `classifier.predict(reading).riskLevel` — trained RF prediction from 7 sensor features

This distinction should be communicated to the user (e.g. "Sensor-computed: UNHEALTHY" vs "ML prediction: UNHEALTHY").

---

## 4. Deployment Format Comparison

### Option A: Generated Kotlin Decision Tree Code

Convert all 200 sklearn trees to Kotlin `TreeNode` data structures, adding a `DoubleArray` to each leaf.

| Criterion | Assessment |
|---|---|
| Prediction equivalence | **Exact** — if implemented correctly |
| Model size (source) | ~3,554 KB of Kotlin source |
| Model size (compiled APK) | ~1,100–1,800 KB additional |
| Android dependencies | **None** — pure Kotlin, no additional libraries |
| Implementation complexity | **Very high** — requires automated code generator; cannot hand-author 52,078 nodes |
| Maintenance | **Very hard** — retraining requires re-running the generator; source is unreadable |
| Offline capability | **Yes** |
| Requires internet | **No** |

**Critical requirement:** `TreeNode.Leaf` must change from `Leaf(classId: Int)` to `Leaf(weights: DoubleArray)` to hold the 5-element weighted probability array. The aggregation in `AirQualityClassifier` must compute mean probabilities, not vote counts.

### Option B: ONNX via ONNX Runtime for Android

Convert the sklearn model to ONNX format and load it at runtime using the ONNX Runtime Android library.

| Criterion | Assessment |
|---|---|
| Prediction equivalence | **Exact** — verified: 100% agreement on 100 test samples |
| Model size (ONNX file) | **3.09 MB** (placed in `assets/`) |
| Android dependencies | `com.microsoft.onnxruntime:onnxruntime-android` (~6 MB AAR) |
| Implementation complexity | **Low** — ~50 lines of Kotlin inference code |
| Maintenance | **Easy** — retrain → re-export ONNX → replace file |
| Offline capability | **Yes** |
| Requires internet | **No** |
| Min Android API | 21+ (ONNX Runtime requirement) |

The ONNX file has already been generated and verified: `final_audit/audit_model_c.onnx` (3.09 MB). ONNX Runtime for Android handles the StandardScaler separately (must be applied in Kotlin before passing to ONNX, since the scaler is not bundled in the ONNX model — it must either be bundled as a second ONNX model or applied with hard-coded Kotlin constants).

### Option C: TensorFlow Lite

Convert sklearn → ONNX → TFLite using the ONNX-to-TFLite converter. TFLite is Google's preferred on-device inference runtime for Android.

| Criterion | Assessment |
|---|---|
| Prediction equivalence | Approximate — the ONNX→TFLite conversion for tree models is lossy in some cases |
| Model size | ~2–4 MB |
| Android dependencies | TFLite is built into Android 8.0+ via NNAPI; library ~1 MB |
| Implementation complexity | **High** — requires ONNX→TFLite conversion which often fails for RF models |
| Maintenance | Moderate |
| Recommendation | **Not recommended** — RF→TFLite conversion is fragile and prediction equivalence is not guaranteed |

### Recommendation: **Option B (ONNX Runtime)**

| Reason | Detail |
|---|---|
| Exact prediction match | Verified 100% agreement on 100 samples |
| Minimal code change | ~50 lines of Kotlin replaces the tree inference |
| Standard toolchain | ONNX Runtime is a production-grade Microsoft/Linux Foundation project |
| Easy to update | Retrain → re-export ONNX → drop in new file |
| No tree encoding | Avoids the 52,078-node code-generation problem entirely |
| APK size impact | +3.09 MB model + ~6 MB ONNX Runtime = ~9 MB total increase |

Option A (generated Kotlin) is a valid fallback if the project has constraints against adding any external dependencies. It requires writing a code generator first.

---

## 5. Equivalence Test Design

A test vector file has been exported to: `final_audit/test_vectors.json`

It contains 48 test cases (up to 10 per present class), with:
- Raw 7-feature values
- Scaled 7-feature values
- Expected predicted class (string)
- Expected class probabilities (by LabelEncoder integer index)

### Kotlin equivalence test procedure

```
For each vector in test_vectors.json:
  1. Load raw_features as DoubleArray
  2. Apply StandardScaler: scaled[i] = (raw[i] - mean[i]) / scale[i]
  3. Run inference (ONNX or generated Kotlin forest)
  4. Decode integer output to class string using label_encoder_mapping
  5. Assert decoded_string == predicted_class

Target: 100% agreement on all 48 test vectors.
```

For probability comparisons (ONNX only):
```
  6. Assert abs(output_proba[i] - expected_proba[i]) < 1e-4 for all 5 classes
```

### Exact agreement guarantee

- **ONNX Runtime**: exact agreement is guaranteed on the 48 exported vectors (verified 100% on 100 samples in this session)
- **Generated Kotlin trees**: exact agreement is guaranteed if leaves store `DoubleArray` weights and aggregation uses `mean-then-argmax` (not majority voting)
- **Majority vote Kotlin**: 5.12% of predictions will differ — this is not acceptable

---

## 6. Files That Would Need Modification

The following files need changes at implementation time. **Nothing has been modified yet.**

### New files to create

| File | Purpose |
|---|---|
| `app/src/main/assets/airguard_model.onnx` | ONNX model file (3.09 MB) |
| `app/src/main/java/com/airguard/ai/ml/OnnxClassifier.kt` | New ONNX-based classifier implementing same `predict()` interface |
| `app/src/main/java/com/airguard/ai/ml/FeatureScaler.kt` | StandardScaler constants and `scale()` function |

### Files to modify

| File | Change required |
|---|---|
| `app/build.gradle.kts` | Add `implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")` |
| `DashboardViewModel.kt` | Swap `AirQualityClassifier()` for `OnnxClassifier()` |
| `AnalysisViewModel.kt` | Same swap |

### Files to leave unchanged initially

| File | Reason |
|---|---|
| `AirQualityClassifier.kt` | Keep as fallback; rename to `LegacyAirQualityClassifier.kt` or deprecate with annotation |
| `DecisionTree.kt` | Keep for now; can be removed after ONNX path is validated |
| `SensorReading.kt` | No changes needed — AQI field stays for display purposes |
| `SensorRepository.kt` | No changes needed |
| `SensorReadingDao.kt` | No changes needed |

---

## 7. Implementation Steps (pending approval)

### Phase 1 — Prepare and validate (Python side, no Android changes)

1. Run the Python–Kotlin equivalence test validator using `test_vectors.json`
2. Confirm the ONNX file already created at `final_audit/audit_model_c.onnx` is the correct artifact
3. Copy `audit_model_c.onnx` → `app/src/main/assets/airguard_model.onnx` (this is a file copy, not a code change)

### Phase 2 — New Kotlin classifier (new files, no existing files touched)

4. Write `FeatureScaler.kt` — 7 hard-coded mean/scale constants and a `scale(raw: DoubleArray): FloatArray` function
5. Write `OnnxClassifier.kt`:
   - Loads ONNX model from assets on first call
   - Applies `FeatureScaler.scale()` to the 7 features
   - Runs ONNX Runtime inference
   - Decodes output integer to `RiskLevel` using the LabelEncoder mapping
   - Returns the same `PredictionResult` type as the existing classifier
   - Takes only 7 features from `SensorReading` (temperature, humidity, pressure, gasKOhms, altitude, dewPoint, heatIndex) — does NOT use `aqi`

### Phase 3 — Swap in ViewModels (minimal change)

6. In `DashboardViewModel.kt`: change `val classifier = AirQualityClassifier()` to `val classifier = OnnxClassifier(app)`
7. In `AnalysisViewModel.kt`: same one-line change

### Phase 4 — Validate on-device

8. Run the 48-vector equivalence test on Android (compare ONNX output against `test_vectors.json`)
9. Confirm all 48 predictions match; confirm probability outputs are within tolerance

### Phase 5 — Cleanup (after validation)

10. Deprecate `AirQualityClassifier.kt` and `DecisionTree.kt` with `@Deprecated` annotations (do not delete until Phase 4 passes)

---

## 8. Remaining Limitations to Disclose

These limitations exist in the trained model and cannot be fixed by changing the deployment format:

1. **VERY_UNHEALTHY cannot be predicted** — zero training samples; model outputs only 5 classes. Android must handle this gracefully (e.g. show "Cannot classify — insufficient data for VERY_UNHEALTHY range").
2. **HAZARDOUS recall = 49.2%** — the model misses ~half of HAZARDOUS conditions; for a safety application this must be disclosed to users.
3. **HAZARDOUS precision = 100%** — when the model predicts HAZARDOUS, it is always correct on the test set.
4. **Training data is from one sensor at one location** — predictions on a device in a different environment may be less accurate.
5. **Labels are AQI-derived, not EPA-validated** — the AQI column in the dataset comes from the sensor's internal calculation, not a certified reference instrument.

---

## 9. Waiting for Approval

**No Android files have been modified.** The following items are ready if you approve:

- `audit_model_c.onnx` exists at `ml_pipeline/experiments/final_audit/audit_model_c.onnx` (3.09 MB, verified)
- `test_vectors.json` exists at `ml_pipeline/experiments/final_audit/test_vectors.json` (48 vectors)
- Scaler constants are confirmed in `audit_config.json`

Awaiting your decision on:
1. **Deployment format** — ONNX Runtime (recommended) or generated Kotlin trees?
2. **New class name** — `OnnxClassifier` or `RfClassifier` or keep the existing `AirQualityClassifier` name and add a constructor parameter?
3. **Dual-mode display** — should the app show both the AQI-derived label and the ML prediction label separately, or replace one with the other?
