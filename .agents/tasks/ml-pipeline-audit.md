# AirGuard AI ML Pipeline — Validation Audit Report

**Date**: Audit performed on saved artifacts in `ml_pipeline/`  
**Scope**: Read-only. No files were modified, no model was retrained.  
**All numbers below come from the loaded `.pkl` artifacts and the original CSV, not from `training_output.txt`.**

---

## 1. Macro-F1 Verification

### Summary
The reported macro-F1 of **0.5642** was computed over **5 present classes** only (VERY_UNHEALTHY had zero samples and was excluded from the LabelEncoder). When all **6 canonical classes** are included with `zero_division=0`, the corrected macro-F1 drops to **0.4701** — a difference of **−0.0941**.

### LabelEncoder Classes (as saved in `airguard_label_encoder.pkl`)
The encoder knows only 5 classes:
```
[0] GOOD
[1] HAZARDOUS
[2] MODERATE
[3] UNHEALTHY
[4] UNHEALTHY_SENSITIVE
```
VERY_UNHEALTHY is absent because it had zero training samples. This is factually correct behaviour, but it means the reported macro metrics were averaged over 5 classes, not 6.

### Recomputed Metrics

| Metric | Reported (5 classes) | Corrected (6 classes) | Difference |
|---|---|---|---|
| Accuracy | 0.8765 | 0.8765 (unchanged) | 0.0000 |
| Macro Precision | 0.5624 | 0.4687 | −0.0937 |
| Macro Recall | 0.6837 | 0.5698 | −0.1139 |
| **Macro F1** | **0.5642** | **0.4701** | **−0.0941** |

### Per-Class Metrics (6-class space, `zero_division=0`)

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| GOOD | 10 | 0.9091 | 1.0000 | 0.9524 |
| MODERATE | 16 | 0.8000 | 0.5000 | 0.6154 |
| UNHEALTHY_SENSITIVE | 42 | 0.1869 | 0.9524 | 0.3125 |
| UNHEALTHY | 3530 | 0.9162 | 0.9663 | 0.9406 |
| **VERY_UNHEALTHY** | **0** | **0.0000** | **0.0000** | **0.0000** |
| **HAZARDOUS** | **360** | **0.0000** | **0.0000** | **0.0000** |

HAZARDOUS has 360 test samples but **precision = 0, recall = 0, F1 = 0**. The model predicts zero HAZARDOUS samples correctly — all 360 HAZARDOUS test records are misclassified (312 as UNHEALTHY, 48 as UNHEALTHY_SENSITIVE).

### Confusion Matrix (6×6)

```
Actual \ Predicted  GOOD  MODERATE  UNHEALTHY_S  UNHEALTHY  VERY_UNH  HAZARDOUS
GOOD                  10         0            0          0         0          0
MODERATE               1         8            7          0         0          0
UNHEALTHY_SENSITIVE    0         2           40          0         0          0
UNHEALTHY              0         0          119       3411         0          0
VERY_UNHEALTHY         0         0            0          0         0          0
HAZARDOUS              0         0           48        312         0          0
```

**Critical finding**: The model never predicts HAZARDOUS. All 360 HAZARDOUS test samples go to UNHEALTHY (312) or UNHEALTHY_SENSITIVE (48). HAZARDOUS was absent from the training set (appeared only after the 80% split boundary, from 2026-05-30 onwards), so the classifier never learned the class.

---

## 2. Chronological Split Audit

### Split Boundary

| | Timestamp |
|---|---|
| Train minimum | 2025-07-26 13:38:48 |
| Train maximum | 2026-05-01 04:03:03 |
| Test minimum | 2026-05-01 04:03:09 |
| Test maximum | 2026-08-25 10:42:29 |

The split uses a row-index boundary (80% of sorted rows), not a calendar-date boundary.

### 2026-05-01 Boundary Overlap

2026-05-01 contains **5,378 total records** that are split mid-day:

| Set | Records from 2026-05-01 | Time range on that day |
|---|---|---|
| Train | 3,060 | 00:00:01 → 04:03:03 |
| Test | 2,318 | 04:03:09 → 07:03:28 |

**Why this is a problem**: Records from the same calendar date that share the same environmental conditions (same diurnal cycle, same weather) appear in both sets. The gap between the last train record and first test record is only **6 seconds**. Any temporal autocorrelation in the sensor readings (temperature/humidity gradually changing through the morning) means these records are not truly independent. This inflates test performance because the model has seen the environmental context of that exact morning.

### Recommended Fix
Use a whole-day cutoff: all records with date **< 2026-05-02** go to training, all records with date **≥ 2026-05-02** go to testing. This keeps complete calendar days together and respects daily periodicity.

```python
cutoff_date = pd.Timestamp("2026-05-02")
df_train = df_dated[df_dated["_datetime"] < cutoff_date]
df_test  = df_dated[df_dated["_datetime"] >= cutoff_date]
```

The resulting split would be approximately 82/18 instead of 80/20, which is acceptable.

---

## 3. Class Distribution Shift

### Train vs Test Class Counts

| Class | Train (N=15832) | % | Test (N=3958) | % |
|---|---|---|---|---|
| GOOD | 125 | 0.79% | 10 | 0.25% |
| MODERATE | 406 | 2.56% | 16 | 0.40% |
| UNHEALTHY_SENSITIVE | 5,833 | 36.84% | 42 | 1.06% |
| UNHEALTHY | 9,468 | 59.80% | 3,530 | 89.19% |
| VERY_UNHEALTHY | 0 | 0.00% | 0 | 0.00% |
| HAZARDOUS | 0 | 0.00% | 360 | 9.10% |

This is a dramatic distribution shift. UNHEALTHY_SENSITIVE drops from 36.8% in training to 1.1% in testing. HAZARDOUS, entirely absent from training, constitutes 9.1% of the test set.

### HAZARDOUS Class First Appearance

| | Timestamp |
|---|---|
| First HAZARDOUS record | **2026-05-30 00:26:54** |
| Last HAZARDOUS record | 2026-08-25 10:42:29 |
| Total HAZARDOUS records | 360 |

All HAZARDOUS records appear after the training cutoff (2026-05-01). The model has zero exposure to this class during training.

### Monthly AQI Distribution

| Month | N | AQI min | AQI max | AQI mean | GOOD | MOD | U_S | UNH | VU | HAZ |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-07 | 2354 | 20 | 80 | 61.1 | 2 | 23 | 2174 | 155 | 0 | 0 |
| 2025-08 | 1963 | 20 | 80 | 62.5 | 24 | 120 | 1402 | 417 | 0 | 0 |
| 2025-09 | 419 | 60 | 80 | 60.0 | 0 | 0 | 418 | 1 | 0 | 0 |
| 2026-01 | 855 | 20 | 80 | 60.3 | 17 | 100 | 593 | 145 | 0 | 0 |
| 2026-02 | 1879 | 10 | 80 | 73.8 | 40 | 83 | 292 | 1464 | 0 | 0 |
| 2026-03 | 4037 | 20 | 80 | 76.6 | 31 | 55 | 487 | 3464 | 0 | 0 |
| 2026-04 | 1265 | 20 | 80 | 71.3 | 11 | 25 | 467 | 762 | 0 | 0 |
| 2026-05 | 6236 | 20 | 100 | 80.7 | 2 | 5 | 17 | 5974 | 0 | 238 |
| 2026-06 | 153 | 20 | 80 | 78.6 | 1 | 2 | 4 | 146 | 0 | 0 |
| 2026-07 | 263 | 20 | 100 | 85.2 | 3 | 5 | 12 | 143 | 0 | 100 |
| 2026-08 | 366 | 20 | 100 | 79.6 | 4 | 4 | 9 | 327 | 0 | 22 |

**Interpretation**: This is a genuine temporal distribution shift in the sensor environment, not a data recording artefact. The AQI maximum does not exceed 80 until May 2026 and does not reach 100 (HAZARDOUS threshold) until May 2026. The mean AQI rises from ~61 in mid-2025 to ~85 in mid-2026. This is consistent with a real-world change in air quality conditions (seasonal, environmental, or equipment drift). The model was trained on a qualitatively different air quality regime than the period it is tested on.

VERY_UNHEALTHY (AQI 81–90) never appears in any month, which is an unexplained gap in the data — the AQI jumps from ≤80 directly to 100 with no intermediate values. This may reflect the sensor's output quantisation or collection methodology, not a genuine absence of that air quality range.

---

## 4. Preprocessing Audit

### Scaler Identity
The saved `airguard_preprocessor.pkl` is a **`sklearn.preprocessing.StandardScaler`**, confirmed by `type(scaler).__name__ == 'StandardScaler'`.

### Feature Order (exact, as stored in `train_model.py` `FEATURES` list)
```
[0] Temperature(°C)
[1] Humidity(%)
[2] Pressure(hPa)
[3] Gas (kOhms)
[4] Altitude(m)
[5] DewPoint(°C)
[6] HeatIndex(°C)
```

### Scaler Parameters (from loaded artifact)

| Index | Feature | mean_ | scale_ |
|---|---|---|---|
| 0 | Temperature(°C) | 31.97325670 | 4.44241592 |
| 1 | Humidity(%) | 65.08602198 | 7.32367635 |
| 2 | Pressure(hPa) | 1002.68610409 | 6.53919221 |
| 3 | Gas (kOhms) | 230.15785750 | 73.16657485 |
| 4 | Altitude(m) | 88.44427741 | 54.94417695 |
| 5 | DewPoint(°C) | 24.96516549 | 4.96462269 |
| 6 | HeatIndex(°C) | 38.65750379 | 8.11243504 |

**Verification**: Re-fitting a fresh StandardScaler on the identical `X_train` matrix produces `mean_` and `scale_` values that match the saved scaler to within `rtol=1e-5`. Confirmed: **the scaler was fitted on training data only**.

### Kotlin Reproduction
The transformation applied to every inference input before the forest must be:
```
scaled_value = (raw_value - mean_) / scale_
```
Applied per feature in the exact order above. The Android implementation **must** use these exact constants and this exact feature order. Any reordering or omission causes systematically wrong predictions.

### Critical Issue: Existing Android App Uses Different Features
The existing `AirQualityClassifier.kt` uses **8 features** including AQI at index 3. The trained `airguard_model.pkl` uses **7 features** without AQI. These are incompatible. The `.pkl` model **cannot be deployed** into the existing `AirQualityClassifier.kt` without redesigning the feature extraction and preprocessing layers.

---

## 5. Model vs Baseline Comparison

| Metric | Value |
|---|---|
| Random Forest accuracy | **87.65%** |
| Majority-class baseline accuracy | **89.19%** |
| Difference | **−1.54 percentage points** |

The model performs **1.54 percentage points worse** than simply predicting UNHEALTHY for every record.

This is not a marginal deficit — it is a sign that the model's errors on minority classes (particularly misclassifying HAZARDOUS as UNHEALTHY or UNHEALTHY_SENSITIVE) cost more in total wrong predictions than the accuracy it gains on those minority classes. The `class_weight='balanced'` hyperparameter causes the model to over-predict minority classes (UNHEALTHY_SENSITIVE) on majority-class test records, reducing overall accuracy.

**Useful predictive value beyond baseline**: Limited. The model correctly identifies GOOD (10/10), most MODERATE (8/16), and most UNHEALTHY (3411/3530). But it completely fails on HAZARDOUS (0/360) — the highest-stakes class — and over-predicts UNHEALTHY_SENSITIVE. For a safety-focused air quality application, failing on HAZARDOUS is a critical deficiency.

---

## 6. Feature Importance Verification

All 7 importance values loaded directly from `clf.feature_importances_`, cross-checked against `feature_importances.csv`. All 7 values match exactly (to 15 significant figures).

| Feature | Importance | CSV Match |
|---|---|---|
| Gas (kOhms) | 0.62799598 | ✓ |
| Temperature(°C) | 0.07930116 | ✓ |
| Humidity(%) | 0.07856102 | ✓ |
| Altitude(m) | 0.06194403 | ✓ |
| Pressure(hPa) | 0.05345232 | ✓ |
| DewPoint(°C) | 0.05331199 | ✓ |
| HeatIndex(°C) | 0.04543350 | ✓ |
| **Sum** | **1.00000000** | — |

The 62.8% Gas importance is authentic — it comes directly from the fitted sklearn model's mean decrease in impurity across all 200 trees.

**Important limitation**: Feature importance measures the variable's role in the trained model's decision boundaries, not physical causation or validated AQI measurement. Gas resistance being important in this model means the sensor's gas reading correlated with AQI-derived labels in this dataset; it does not mean gas resistance is a physically validated AQI predictor.

---

## 7. Model Export Readiness

### Trained Model Parameters

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth (param) | None (unlimited) |
| max_depth (actual, min across trees) | 9 |
| max_depth (actual, max across trees) | 23 |
| max_depth (actual, mean across trees) | 15.0 |
| Leaves per tree (min) | 40 |
| Leaves per tree (max) | 207 |
| Leaves per tree (mean) | 115.8 |
| Nodes per tree (mean) | 230.5 |
| **Total nodes across all 200 trees** | **46,108** |
| n_features_in_ | 7 |
| n_classes_ (trained) | 4 (VERY_UNHEALTHY and HAZARDOUS absent from training) |

### Class Order in Trained Model
`clf.classes_` stores encoded indices mapping to these strings via `le.inverse_transform`:
```
[0] → GOOD
[1] → MODERATE   (encoded index 2 in LabelEncoder)
[2] → UNHEALTHY  (encoded index 3 in LabelEncoder)
[3] → UNHEALTHY_SENSITIVE  (encoded index 4 in LabelEncoder)
```
Note: `clf.classes_` is `[0, 2, 3, 4]` (the encoded integer indices). HAZARDOUS (encoder index 1) and VERY_UNHEALTHY (not in encoder) were never seen during training. The vote array at every leaf is of size **4**, not 6.

### Internal Tree Structure
Each tree node uses float64 thresholds on scaled features (post-StandardScaler). Leaf nodes store a `value` array of shape `(1, n_classes)` = `(1, 4)` containing **float weighted class counts** (not integers), because `class_weight='balanced'` scales class counts by their inverse frequency.

**Example from Tree 0, Node 8** (a leaf):  
`votes=[0, 0, 0, 1]` → UNHEALTHY class (class index 3 in the trained model's 4-class space).

### Compatibility with Existing Kotlin `DecisionTree.kt`

The existing implementation (`c:\Users\ADMIN\AppData\AirGuardAI\app\src\main\java\com\airguard\ai\ml\DecisionTree.kt`) defines:
- `TreeNode.Split(featureIndex, threshold, left, right)` — matches sklearn's split node exactly (feature index, float threshold, left child for `≤`, right child for `>`)
- `TreeNode.Leaf(classId: Int)` — stores a **single integer class ID**

**Problem: `TreeNode.Leaf` is incompatible with sklearn's vote aggregation.**

sklearn's Random Forest aggregation works as follows:
1. For each tree, traverse to a leaf and retrieve the **float vote array** `(n_classes,)`.
2. **Sum** all vote arrays across all 200 trees element-wise: `total_votes[c] += tree_leaf_votes[c]`
3. `argmax(total_votes)` gives the final prediction.

The current Kotlin `DecisionTreeEvaluator.predict` returns a single `classId` per tree (the argmax of that tree's leaf). `AirQualityClassifier.kt` then tallies integer votes across trees. This is correct only when leaf vote arrays have a single non-zero entry (i.e., pure leaves). With `min_samples_leaf=2` and `class_weight='balanced'`, many leaves contain mixed float-weighted votes. For those leaves, `argmax(leaf_votes)` and `argmax(sum_of_all_leaf_votes)` can disagree.

**Consequence**: the Kotlin forest will produce **different predictions from the Python model** on ambiguous boundary cases. This is a correctness gap, not just a performance gap.

### Implementation Requirements for Exact Match

To exactly reproduce sklearn predictions in Kotlin:

| Requirement | Detail |
|---|---|
| **(a) StandardScaler must precede the forest** | Apply `(raw - mean_) / scale_` to all 7 features using the exact constants in Section 4. |
| **(b) Feature order must match exactly** | Temperature, Humidity, Pressure, Gas, Altitude, DewPoint, HeatIndex — indices 0–6. |
| **(c) Leaf must store a DoubleArray** | Change `TreeNode.Leaf(classId: Int)` to `TreeNode.Leaf(votes: DoubleArray)` of size `n_classes_`. |
| **(d) Aggregation must sum vote arrays** | Accumulate `DoubleArray(4)` across all 200 trees, then `argmax` once at the end. |
| **(e) Class label mapping** | The 4-class argmax maps to: 0→GOOD, 1→MODERATE, 2→UNHEALTHY, 3→UNHEALTHY_SENSITIVE. HAZARDOUS and VERY_UNHEALTHY can never be predicted. |
| **(f) NaN handling** | sklearn RF errors on NaN inputs. Android must validate that all 7 sensor readings are finite before calling the model. |
| **(g) Tie-breaking** | On argmax ties, the lower index wins (numpy convention). Class order above determines which class wins a tie. |
| **(h) Total node count** | Encoding all 200 trees as Kotlin data structures requires 46,108 nodes total. This is a large but manageable amount of static Kotlin code (~900KB–2MB depending on formatting). |

### Note on Existing `AirQualityClassifier.kt`
The current file uses hand-authored trees with 15 decision trees and AQI as a feature at index 3. It is **not** derived from the trained `airguard_model.pkl`. It is a manually constructed approximation. Any Android integration of the trained model requires replacing or supplementing this file, not just adding preprocessing.

---

## 8. Final Verdict

### Is the model scientifically suitable for the research paper?

**Conditionally yes, with mandatory caveats.**

The pipeline is methodologically sound in its core design:
- AQI is correctly excluded from input features.
- The chronological split is correctly ordered (earliest records to training, latest to testing).
- The scaler is fitted on training data only.
- Feature importances are authentic.
- Class distribution and limitations are honestly reported.

However, the paper **must correct** these factual issues:
1. Macro-F1 of **0.5642** must be replaced with the 6-class corrected figure of **0.4701**.
2. The model must be described as **unable to predict HAZARDOUS** (the class with the highest health impact), because it was absent from training.
3. The model must be reported as performing **1.54 percentage points below the majority-class baseline**.
4. The distribution shift (HAZARDOUS only appearing in the last 20% of the timeline) must be discussed as a fundamental limitation of this dataset's temporal structure.
5. VERY_UNHEALTHY's complete absence from the data must be explicitly noted as a gap — it is impossible to know whether this reflects true environmental conditions or sensor quantisation.

### Is the model technically ready for Android integration right now?

**No.**

The following blockers prevent direct Android integration:

| # | Blocker | Severity |
|---|---|---|
| 1 | `TreeNode.Leaf` stores `classId: Int` but needs `votes: DoubleArray` for exact sklearn matching | Critical |
| 2 | Existing `AirQualityClassifier.kt` uses 8 features including AQI; trained model uses 7 features without AQI — incompatible interfaces | Critical |
| 3 | HAZARDOUS class can never be predicted by the trained model (not in training data) | Critical for safety |
| 4 | 46,108 tree nodes must be encoded as Kotlin data structures — feasible but requires automated code generation, not manual authoring | Significant effort |
| 5 | StandardScaler constants must be hard-coded in a Kotlin preprocessing step with exact values from Section 4 | Required |
| 6 | NaN sensor value guard must be added before model inference | Required |

### What must be fixed first (ordered)

1. **Re-run training with HAZARDOUS in the training set.** The whole-day cutoff fix (Section 2) will move 2,318 records from 2026-05-01 into training, but HAZARDOUS still won't appear until 2026-05-30. The dataset must be used with a later cutoff (e.g., 70/30 split, or a fixed date of 2026-05-01) so that HAZARDOUS records fall in both train and test. If the dataset cannot provide HAZARDOUS training samples, the model fundamentally cannot classify that risk level.

2. **Correct the reported macro-F1** in any paper or presentation to the 6-class figure (0.4701) and disclose the HAZARDOUS failure explicitly.

3. **Fix the split boundary** to use whole calendar days (cutoff: `< 2026-05-02` for train) to eliminate the 6-second intra-day boundary overlap.

4. **For Android integration**: Modify `TreeNode.Leaf` to carry `DoubleArray` vote weights and update `DecisionTreeEvaluator` to return the summed-vote argmax across all trees. Write a code generator (Python script) that walks all 200 trees and emits the Kotlin `TreeNode` structure — do not hand-author 46,108 nodes.

5. **Add StandardScaler preprocessing** as a Kotlin function using the exact constants from Section 4, applied before passing features to the forest.

6. **Remove AQI from `AirQualityClassifier.kt`'s `extractFeatures`** (or create a new classifier class) so the feature vector matches the trained model's 7-feature input.

7. **Add NaN guards** in the Android sensor reading pipeline before model inference.
