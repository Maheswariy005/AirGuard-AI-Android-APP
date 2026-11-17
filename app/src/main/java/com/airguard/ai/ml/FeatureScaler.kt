package com.airguard.ai.ml

/**
 * StandardScaler constants for the verified RF Model C (cutoff 2026-06-01).
 *
 * These values were fitted on the training set only (19,008 rows,
 * 2025-07-26 to 2026-05-31) and must not be recomputed from test data.
 *
 * Scaling formula: scaled[i] = (raw[i] - mean[i]) / scale[i]
 *
 * Feature order (must match the ONNX model input exactly):
 *   [0] Temperature (°C)
 *   [1] Humidity (%)
 *   [2] Pressure (hPa)
 *   [3] Gas (kOhms)
 *   [4] Altitude (m)
 *   [5] Dew Point (°C)
 *   [6] Heat Index (°C)
 *
 * AQI is NOT a feature and must never be passed to this scaler or the model.
 *
 * Source: ml_pipeline/experiments/final_audit/audit_config.json
 */
object FeatureScaler {

    val FEATURE_NAMES = arrayOf(
        "Temperature (°C)",   // index 0
        "Humidity (%)",       // index 1
        "Pressure (hPa)",     // index 2
        "Gas (kOhms)",        // index 3
        "Altitude (m)",       // index 4
        "Dew Point (°C)",     // index 5
        "Heat Index (°C)"     // index 6
    )
    const val FEATURE_COUNT = 7

    /**
     * Feature importances from the verified ONNX Random Forest model
     * (RF class_weight='balanced', 200 trees, random_state=42, cutoff 2026-06-01).
     *
     * Source: ml_pipeline/experiments/final_audit/feature_importances.csv
     * These are the actual sklearn feature_importances_ values, not approximations.
     * AQI is NOT included — it was never a model input.
     *
     * Order matches FEATURE_NAMES above.
     */
    val FEATURE_IMPORTANCES = doubleArrayOf(
        0.065123,   // Temperature
        0.124538,   // Humidity
        0.071816,   // Pressure
        0.541908,   // Gas (kOhms) — dominant feature
        0.085349,   // Altitude
        0.056730,   // Dew Point
        0.054536    // Heat Index
    )

    // Fitted on training data only. Do not update without retraining.
    private val MEANS = doubleArrayOf(
        32.404256,    // Temperature
        65.030945,    // Humidity
        1002.640914,  // Pressure
        253.995597,   // Gas
        88.799590,    // Altitude
        25.388473,    // Dew Point
        39.360001     // Heat Index
    )

    private val SCALES = doubleArrayOf(
        4.205421,   // Temperature
        7.397750,   // Humidity
        5.991025,   // Pressure
        89.545656,  // Gas
        50.336149,  // Altitude
        4.758210,   // Dew Point
        7.739075    // Heat Index
    )

    /**
     * Scale a raw 7-feature array into a Float32 array for ONNX inference.
     *
     * @param raw DoubleArray of length 7 in feature order above.
     * @return FloatArray of length 7, ready for ONNX input.
     * @throws IllegalArgumentException if raw.size != FEATURE_COUNT
     * @throws IllegalArgumentException if any value is NaN or infinite
     */
    fun scale(raw: DoubleArray): FloatArray {
        require(raw.size == FEATURE_COUNT) {
            "Expected $FEATURE_COUNT features, got ${raw.size}"
        }
        for (i in raw.indices) {
            require(raw[i].isFinite()) {
                "Feature ${FEATURE_NAMES[i]} is not finite: ${raw[i]}"
            }
        }
        return FloatArray(FEATURE_COUNT) { i ->
            ((raw[i] - MEANS[i]) / SCALES[i]).toFloat()
        }
    }
}
