# AirGuard AI — Chronological Split Investigation & Model Comparison Report

**All numbers in this report come from actual Python execution on the saved artifacts and original CSV.**  
**No results are fabricated or extrapolated.**  
No Android/Kotlin files were modified. The original CSV and existing `ml_pipeline/` artifacts are unchanged.

---

## 1. HAZARDOUS Sample Distribution by Date

### Dataset after cleaning
| Step | Rows |
|---|---|
| Raw CSV | 19,896 |
| After deduplication | 19,894 |
| After dropping NaN in features/AQI | 19,894 |
| After dropping rows with unparseable datetime | 19,790 |

### HAZARDOUS samples by date

| Date | Total samples | HAZARDOUS | HAZ% | AQI min | AQI max |
|---|---:|---:|---:|---:|---:|
| 2026-05-30 | 781 | 238 | 30.5% | 100 | 100 |
| 2026-07-14 | 185 | 51 | 27.6% | 100 | 100 |
| 2026-07-21 | 78 | 49 | 62.8% | 100 | 100 |
| 2026-08-25 | 24 | 22 | 91.7% | 100 | 100 |

| Stat | Value |
|---|---|
| First HAZARDOUS record | 2026-05-30 00:26:54 |
| Last HAZARDOUS record | 2026-08-25 10:42:29 |
| Total HAZARDOUS samples | 360 |
| Distinct dates with HAZARDOUS | **4** |

**Critical finding:** HAZARDOUS only appears on 4 calendar dates, all between May 2026 and August 2026. All HAZARDOUS records have AQI = 100 exactly (the maximum in the dataset). There are no AQI values between 81 and 90, and no AQI values between 91 and 99. This is consistent with sensor output quantisation or recording methodology, not a continuous air quality distribution.

### Monthly class distribution

| Month | GOOD | MODERATE | UNH_SENS | UNHEALTHY | VERY_UNH | HAZARDOUS |
|---|---:|---:|---:|---:|---:|---:|
| 2025-07 | 2 | 23 | 2174 | 155 | 0 | 0 |
| 2025-08 | 24 | 120 | 1402 | 417 | 0 | 0 |
| 2025-09 | 0 | 0 | 418 | 1 | 0 | 0 |
| 2026-01 | 17 | 100 | 593 | 145 | 0 | 0 |
| 2026-02 | 40 | 83 | 292 | 1,464 | 0 | 0 |
| 2026-03 | 31 | 55 | 487 | 3,464 | 0 | 0 |
| 2026-04 | 11 | 25 | 467 | 762 | 0 | 0 |
| 2026-05 | 2 | 5 | 17 | 5,974 | 0 | 238 |
| 2026-06 | 1 | 2 | 4 | 146 | 0 | 0 |
| 2026-07 | 3 | 5 | 12 | 143 | 0 | 100 |
| 2026-08 | 4 | 4 | 9 | 327 | 0 | 22 |

### Can HAZARDOUS support a chronological train/test split?

Barely. With only 4 source dates, the earliest split that places HAZARDOUS in both sets requires putting 2026-05-30 (238 samples) into training and keeping the remaining 3 dates (122 samples) in testing. This gives a 2026-06-01 cutoff with:
- Training HAZARDOUS: 238
- Test HAZARDOUS: 122

Any cutoff before 2026-05-30 gives zero HAZARDOUS training samples, making HAZARDOUS completely unpredictable.

---

## 2. VERY_UNHEALTHY Confirmation

**VERY_UNHEALTHY total samples across the entire dataset: 0**

There are zero AQI values in the range 81–90 anywhere in the dataset. The AQI column contains only values: 10, 20, 40, 60, 80, 100. This appears to be a sensor quantisation artifact — the BME680/AeroSense system reports AQI in fixed 20-unit steps.

**How this must be disclosed in the research paper:**

> "The AeroSense Device 2 sensor system produces AQI values in discrete steps of 20 (10, 20, 40, 60, 80, 100). No readings in the 81–90 range (VERY_UNHEALTHY threshold) were recorded in the entire dataset of 19,790 temporally valid samples. The trained classifier cannot predict the VERY_UNHEALTHY class. This is a fundamental characteristic of the sensor's output, not a limitation of the classification algorithm. Any deployment of this model must note that VERY_UNHEALTHY cannot be detected with this sensor configuration."

---

## 3. Candidate Split Evaluation

All splits use whole calendar days. No date appears in both training and testing.

| Cutoff | Train | Test | Train HAZ | Test HAZ | HAZ in both (≥30) | Viable |
|---|---:|---:|---:|---:|:---:|:---:|
| 2026-04-01 | 11,507 | 8,283 | 0 | 360 | NO | NO |
| 2026-04-15 | 11,507 | 8,283 | 0 | 360 | NO | NO |
| 2026-05-01 | 12,772 | 7,018 | 0 | 360 | NO | NO |
| 2026-05-15 | 18,150 | 1,640 | 0 | 360 | NO | NO |
| 2026-05-29 | 18,150 | 1,640 | 0 | 360 | NO | NO |
| **2026-06-01** | **19,008** | **782** | **238** | **122** | **YES** | **YES** |
| 2026-07-01 | 19,161 | 629 | 238 | 122 | YES | YES |
| 2026-07-14 | 19,161 | 629 | 238 | 122 | YES | YES |

**Selected split: cutoff = 2026-06-01**

Rationale:
- The only way to get HAZARDOUS in both sets is to include 2026-05-30 in training. Any cutoff ≥ 2026-06-01 achieves this.
- 2026-06-01 is selected over 2026-07-01 because it provides more test data (782 vs 629 rows), giving more stable test-set estimates.
- The cutoff was not chosen to maximise accuracy. It was chosen as the earliest date that makes the split viable.
- Train/test ratio is approximately 96/4 — an unusually skewed split, but unavoidable given that HAZARDOUS only appears in the last 2.4% of the timeline.

**Important limitation:** The 96/4 split means the test set has only 782 samples. Per-class test support is small for minority classes: GOOD (8), MODERATE (11), UNHEALTHY_SENSITIVE (25), HAZARDOUS (122). Metric estimates for these classes have high variance.

---

## 4. Model Experiment Results

**Split:** cutoff = 2026-06-01  
**Train:** 19,008 rows (2025-07-26 → 2026-05-31)  
**Test:** 782 rows (2026-06-01 → 2026-08-25)  
**Date overlap:** 0 dates (clean)  
**Preprocessing:** StandardScaler fitted on training data only, applied to test without refitting.

### 4a. Comparison Table (all 6 classes, zero_division=0)

| Experiment | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | HAZ Recall |
|---|---:|---:|---:|---:|---:|---:|
| A. Majority-class baseline | 78.77% | 0.1313 | 0.1667 | 0.1469 | 0.6942 | 0.000 |
| B. RF class_weight=None | **91.94%** | 0.6702 | 0.7306 | 0.6648 | **0.9265** | 0.566 |
| C. RF class_weight='balanced' | 90.54% | 0.7165 | 0.7113 | **0.6966** | 0.8946 | 0.492 |
| D. RF class_weight='balanced_subsample' | 87.60% | **0.7273** | 0.6841 | 0.6702 | 0.8494 | 0.287 |

**All three RF models beat the majority-class baseline** on accuracy, macro F1, and HAZARDOUS recall.

### 4b. Per-class metrics — Best model: C. RF class_weight='balanced' (highest macro F1 = 0.6966)

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| GOOD | 8 | 0.8889 | 1.0000 | 0.9412 |
| MODERATE | 11 | 0.7692 | 0.9091 | 0.8333 |
| UNHEALTHY_SENSITIVE | 25 | 0.7333 | 0.8800 | 0.8000 |
| UNHEALTHY | 616 | 0.9075 | 0.9870 | 0.9456 |
| VERY_UNHEALTHY | 0 | 0.0000 | 0.0000 | 0.0000 |
| HAZARDOUS | 122 | 1.0000 | 0.4918 | 0.6593 |

### 4c. Confusion Matrix — Model C (RF balanced), 6×6

Rows = actual class, columns = predicted class.

| Actual \ Predicted | GOOD | MODERATE | UNH_S | UNHEALTHY | VERY_UNH | HAZARDOUS |
|---|---:|---:|---:|---:|---:|---:|
| GOOD | **8** | 0 | 0 | 0 | 0 | 0 |
| MODERATE | 0 | **10** | 0 | 1 | 0 | 0 |
| UNHEALTHY_SENSITIVE | 0 | 1 | **22** | 2 | 0 | 0 |
| UNHEALTHY | 0 | 0 | 8 | **608** | 0 | 0 |
| VERY_UNHEALTHY | 0 | 0 | 0 | 0 | 0 | 0 |
| HAZARDOUS | 0 | 0 | 0 | 62 | 0 | **60** |

HAZARDOUS: 60 of 122 correctly predicted. 62 HAZARDOUS samples are misclassified as UNHEALTHY.

### 4d. Feature importances — Model C (RF balanced)

| Feature | Importance |
|---|---:|
| Gas (kOhms) | 0.541908 |
| Humidity(%) | 0.124538 |
| Altitude(m) | 0.085349 |
| Pressure(hPa) | 0.071816 |
| Temperature(°C) | 0.065123 |
| DewPoint(°C) | 0.056730 |
| HeatIndex(°C) | 0.054536 |
| **Sum** | **1.000000** |

Gas resistance remains the dominant feature (54.2%), down from 62.8% in the original model, with Humidity gaining importance (12.5% vs 7.9%). This shift occurs because HAZARDOUS is now in the training data and the model must distinguish it from UNHEALTHY.

---

## 5. Evaluation Reliability Check

| Check | Result |
|---|---|
| StandardScaler fitted on training data only | Confirmed |
| Scaler not refit on test data | Confirmed |
| Feature order unchanged (7 features, same names) | Confirmed |
| No date shared between train and test | Confirmed (0 overlap) |
| No random split used | Confirmed — strict calendar-day cutoff |
| Duplicate records removed before split | Confirmed (2 removed) |
| NaN rows removed before split | Confirmed (104 datetime-less rows excluded) |
| No hyperparameter tuning on test set | Confirmed — fixed RF_BASE params, no grid search |

**Scaler parameters (new, fit on 2026-06-01 split training data):**

| Feature | mean_ | scale_ |
|---|---:|---:|
| Temperature(°C) | 32.4043 | 4.2054 |
| Humidity(%) | 65.0309 | 7.3978 |
| Pressure(hPa) | 1002.6409 | 5.9910 |
| Gas (kOhms) | 253.9956 | 89.5457 |
| Altitude(m) | 88.7996 | 50.3361 |
| DewPoint(°C) | 25.3885 | 4.7582 |
| HeatIndex(°C) | 39.3600 | 7.7391 |

---

## 6. Temporal Distribution Shift

The training data (Jul 2025 – May 2026) and test data (Jun 2026 – Aug 2026) represent qualitatively different air quality regimes:

- **Training period**: UNHEALTHY_SENSITIVE accounts for 30.8% of training samples (5,850 rows)
- **Test period**: UNHEALTHY_SENSITIVE is only 3.2% of test samples (25 rows)
- **HAZARDOUS**: 1.25% of training, 15.6% of test

The model is being tested on a period where air quality was substantially worse than what it was primarily trained on. The 96/4 split is not a methodological choice — it is a consequence of the data's temporal structure. Any earlier cutoff eliminates HAZARDOUS from training entirely.

This is a **genuine limitation**: the model was trained on a predominantly UNHEALTHY_SENSITIVE environment and is evaluated on a predominantly UNHEALTHY/HAZARDOUS environment. The distribution shift is real, not an artifact of the split strategy.

---

## 7. Final Recommendation

### Best defensible split
**Cutoff: 2026-06-01** (train < 2026-06-01, test ≥ 2026-06-01)
- Only split that places HAZARDOUS in both sets with ≥30 samples each
- Zero date overlap between sets
- Uses whole calendar days, no intra-day boundary

### Model comparison summary
All three RF models significantly outperform the majority-class baseline (78.77%) in accuracy, macro F1, and HAZARDOUS recall. This is an improvement over the original pipeline where the model was 1.54 pp below baseline with zero HAZARDOUS recall.

**Best model: RF class_weight='balanced'**
- Accuracy: 90.54% (+11.77 pp over baseline)
- Macro F1: 0.6966 (up from 0.4701 in original audit)
- HAZARDOUS recall: 0.492 (up from 0.000)
- HAZARDOUS precision: 1.000 (all HAZARDOUS predictions are correct)
- When the model predicts HAZARDOUS, it is always right; it misses ~half of actual HAZARDOUS events

### HAZARDOUS recognition status
Significantly improved but not solved. 60 of 122 HAZARDOUS test samples are correctly identified. 62 are misclassified as UNHEALTHY. This is partly a consequence of having only one HAZARDOUS training date (2026-05-30) — the model has limited HAZARDOUS diversity to learn from.

### Dataset limitations (remaining)
1. **VERY_UNHEALTHY** has zero samples due to sensor quantisation (AQI only takes values 10, 20, 40, 60, 80, 100). Cannot be predicted.
2. **HAZARDOUS comes from only one training date** (2026-05-30). The model may have learned that date's specific conditions rather than general HAZARDOUS patterns.
3. **96/4 train/test split** is forced by the data's temporal structure — HAZARDOUS only emerges in the last months of the dataset. The test set (782 rows) is small.
4. **Small test-set support for minority classes**: GOOD (8), MODERATE (11), UNHEALTHY_SENSITIVE (25). Metrics for these classes are estimates with high variance.
5. **Single-device, single-location** data. Generalisation to other environments is unverified.
6. **No VERY_UNHEALTHY** in any split, ever. Disclosure required in paper.

### Recommendation

**Proceed with the RF balanced model (cutoff 2026-06-01) for the research paper, with explicit disclosure of all limitations.**

The model now produces scientifically defensible results:
- It beats the baseline by >11 pp on accuracy
- It correctly identifies HAZARDOUS in ~49% of cases (vs 0% before)
- HAZARDOUS precision is 1.0 — no false HAZARDOUS alarms
- All metrics are computed over all 6 canonical classes with zero_division=0

For the **paper**, this split and these results are the honest evaluation to report.

For **Android integration**, the model is not yet ready. The existing blockers from the audit remain:
- `TreeNode.Leaf` must store vote weights, not a single class ID
- 46,108 total nodes require automated code generation
- StandardScaler must be reproduced in Kotlin using the new scaler constants above
- VERY_UNHEALTHY can never be predicted and must be handled gracefully in the UI

**Do not collect synthetic data or oversample to fix VERY_UNHEALTHY** — the sensor physically does not produce AQI values in the 81–90 range. The classification approach for VERY_UNHEALTHY must be reconsidered if that category is required, potentially requiring a different sensor or manual label override.
