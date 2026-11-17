package com.airguard.ai.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.airguard.ai.data.RiskLevel
import com.airguard.ai.data.SensorReading
import java.nio.FloatBuffer

/**
 * On-device Random Forest classifier using ONNX Runtime.
 *
 * Model: RF class_weight='balanced', 200 trees, random_state=42.
 * Trained on AeroSense Device 2 sensor data (chronological split 2026-06-01).
 * Verified: 100% prediction agreement with Python sklearn on 782 test samples.
 *
 * Input: 7 sensor features ONLY — AQI is NOT an input.
 *   [0] Temperature (°C)
 *   [1] Humidity (%)
 *   [2] Pressure (hPa)
 *   [3] Gas resistance (kOhms)
 *   [4] Altitude (m)
 *   [5] Dew Point (°C)
 *   [6] Heat Index (°C)
 *
 * Output: 5 classes (LabelEncoder alphabetical order):
 *   0 -> GOOD
 *   1 -> HAZARDOUS
 *   2 -> MODERATE
 *   3 -> UNHEALTHY
 *   4 -> UNHEALTHY_SENSITIVE
 *
 * ONNX output format: exported with ZipMap=False.
 *   - output "label"        : int64 tensor [n]       — predicted class integer
 *   - output "probabilities": float32 tensor [n, 5]  — class probs in LabelEncoder order
 *
 * VERY_UNHEALTHY is NOT predicted — the training dataset contained zero
 * samples with AQI in the 81-90 range. If the model returns no VERY_UNHEALTHY
 * prediction, this is expected behaviour, not a bug.
 *
 * HAZARDOUS recall on the test set is 49.2% (precision 100%). The model
 * is conservative: when it predicts HAZARDOUS it is correct, but it misses
 * approximately half of HAZARDOUS events. This limitation must be communicated
 * to users.
 *
 * This class is NOT thread-safe. Create one instance per ViewModel or
 * protect with a mutex for concurrent use.
 */
class OnnxClassifier(private val context: Context) {

    companion object {
        private const val TAG = "OnnxClassifier"
        private const val MODEL_ASSET = "airguard_model.onnx"

        /**
         * Maps ONNX integer output (LabelEncoder alphabetical) to RiskLevel.
         * Must match the LabelEncoder used during training exactly.
         *
         *   0 -> GOOD
         *   1 -> HAZARDOUS
         *   2 -> MODERATE
         *   3 -> UNHEALTHY
         *   4 -> UNHEALTHY_SENSITIVE
         *
         * VERY_UNHEALTHY (index 5) is never output by this model.
         */
        private val INT_TO_RISK = mapOf(
            0 to RiskLevel.GOOD,
            1 to RiskLevel.HAZARDOUS,
            2 to RiskLevel.MODERATE,
            3 to RiskLevel.UNHEALTHY,
            4 to RiskLevel.UNHEALTHY_SENSITIVE
        )

        /**
         * Canonical class names in display order (matches RiskLevel.entries order).
         * Used for probability array indexing in PredictionResult.
         * Note: probabilities come out in LabelEncoder order (0=GOOD,1=HAZ,2=MOD,3=UNH,4=UNH_S).
         */
        private val CANONICAL_ORDER = listOf(
            RiskLevel.GOOD,
            RiskLevel.MODERATE,
            RiskLevel.UNHEALTHY_SENSITIVE,
            RiskLevel.UNHEALTHY,
            RiskLevel.VERY_UNHEALTHY,
            RiskLevel.HAZARDOUS
        )
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    /**
     * Lazily initialise the ONNX session on first call.
     * Loads the model from assets — do not call on the main thread.
     */
    @Synchronized
    private fun ensureSession() {
        if (session != null) return
        try {
            Log.i(TAG, "Loading ONNX model from assets/$MODEL_ASSET")
            val ortEnv = OrtEnvironment.getEnvironment()
            val bytes = context.assets.open(MODEL_ASSET).readBytes()
            val opts = OrtSession.SessionOptions()
            session = ortEnv.createSession(bytes, opts)
            env = ortEnv
            Log.i(TAG, "ONNX session created. " +
                "Inputs: ${session!!.inputNames}  Outputs: ${session!!.outputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ONNX session", e)
            throw RuntimeException("OnnxClassifier: could not load model from assets/$MODEL_ASSET", e)
        }
    }

    /**
     * Classify a sensor reading using the trained RF model.
     *
     * AQI from the reading is NOT passed to the model.
     * The returned [MlPredictionResult.aqiDerivedRisk] is set separately
     * from [SensorReading.riskLevel] so the caller can display both.
     *
     * @throws IllegalArgumentException if any sensor value is NaN or infinite.
     * @throws RuntimeException if the ONNX model cannot be loaded.
     */
    fun predict(reading: SensorReading): MlPredictionResult {
        val rawFeatures = extractFeatures(reading)
        return predictFromFeatures(rawFeatures, reading.riskLevel)
    }

    /**
     * Classify from a raw 7-element feature array (no SensorReading required).
     *
     * This method is @Synchronized on the OnnxClassifier instance. Combined
     * with @Synchronized on ensureSession() and close(), this guarantees:
     *
     *   - A prediction that has started will complete before close() runs.
     *   - close() will not run while a prediction is in progress.
     *   - A prediction called after close() will safely re-open the session
     *     via ensureSession() (single-use lifecycle is not enforced; the
     *     ViewModel can call predict() again if recreated).
     *   - Concurrent predictions from multiple threads are serialized
     *     (OnnxClassifier is documented as not thread-safe; this provides
     *     a safety net but callers should still use a single thread).
     *
     * Resource closure: inputTensor and results are closed in finally blocks
     * so they are released even if sess.run() or probability extraction throws.
     *
     * @param rawFeatures DoubleArray[7] in order:
     *   temperature, humidity, pressure, gasKOhms, altitude, dewPoint, heatIndex
     * @param aqiDerivedRisk the AQI-derived classification to carry through
     *   for display alongside the ML prediction (not used as input).
     */
    @Synchronized
    fun predictFromFeatures(rawFeatures: DoubleArray, aqiDerivedRisk: RiskLevel): MlPredictionResult {
        ensureSession()
        // Capture local references so close() nulling env/session mid-call is harmless
        // (ensureSession() just returned with the lock held, so session/env are non-null here)
        val sess  = session  ?: throw RuntimeException("ONNX session not initialised")
        val ortEnv = env     ?: throw RuntimeException("ORT environment not initialised")

        // Apply StandardScaler (training-data statistics, no test leakage)
        val scaled = FeatureScaler.scale(rawFeatures)

        // Build ONNX input tensor: shape [1, 7], float32
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(scaled),
            longArrayOf(1L, FeatureScaler.FEATURE_COUNT.toLong())
        )

        // Run inference; guarantee inputTensor is always closed
        val results = try {
            sess.run(mapOf("float_input" to inputTensor))
        } finally {
            // Closed here whether run() succeeds or throws
            inputTensor.close()
        }

        // Extract outputs; guarantee results are always closed
        //
        // ONNX output layout (ZipMap=False export):
        //   output 0 "label"        : int64 tensor [1]        — predicted class integer
        //   output 1 "probabilities": float32 tensor [1, 5]   — class probabilities in
        //                              LabelEncoder alphabetical order:
        //                              [GOOD, HAZARDOUS, MODERATE, UNHEALTHY, UNHEALTHY_SENSITIVE]
        val predictedInt: Int
        val probaArray: FloatArray  // length 5, LabelEncoder order
        try {
            // Output 0: predicted class as int64
            @Suppress("UNCHECKED_CAST")
            val labelArray = results[0].value as LongArray
            predictedInt = labelArray[0].toInt()

            // Output 1: probabilities as float32 tensor [1, n_classes]
            // On Android, OrtSession returns this as a flat FloatArray
            @Suppress("UNCHECKED_CAST")
            val rawProba = results[1].value  // Array<FloatArray> or FloatArray depending on ORT version
            probaArray = when (rawProba) {
                is Array<*> -> {
                    // Array<FloatArray> — 2D case: shape [1, 5]
                    @Suppress("UNCHECKED_CAST")
                    (rawProba as Array<FloatArray>)[0]
                }
                is FloatArray -> rawProba  // flat 1D case (some ORT versions)
                else -> throw RuntimeException(
                    "Unexpected probability output type: ${rawProba?.javaClass?.name}"
                )
            }
        } finally {
            results.close()
        }

        // Decode predicted class (default to UNHEALTHY if unknown int — safety fallback)
        val predictedRisk = INT_TO_RISK[predictedInt]
            ?: run {
                Log.w(TAG, "Unknown ONNX output int $predictedInt; defaulting to UNHEALTHY")
                RiskLevel.UNHEALTHY
            }

        // Build canonical-order probability list (6 slots, VERY_UNHEALTHY always 0.0)
        // LabelEncoder alphabetical order in probaArray:
        //   index 0 -> GOOD, 1 -> HAZARDOUS, 2 -> MODERATE,
        //   3 -> UNHEALTHY, 4 -> UNHEALTHY_SENSITIVE
        val probByRisk: Map<RiskLevel, Double> = mapOf(
            RiskLevel.GOOD                to (probaArray.getOrElse(0) { 0f }.toDouble()),
            RiskLevel.MODERATE            to (probaArray.getOrElse(2) { 0f }.toDouble()),
            RiskLevel.UNHEALTHY_SENSITIVE to (probaArray.getOrElse(4) { 0f }.toDouble()),
            RiskLevel.UNHEALTHY           to (probaArray.getOrElse(3) { 0f }.toDouble()),
            RiskLevel.VERY_UNHEALTHY      to 0.0,  // never predicted; always 0
            RiskLevel.HAZARDOUS           to (probaArray.getOrElse(1) { 0f }.toDouble()),
        )

        val mlProbabilities = CANONICAL_ORDER.map { probByRisk[it] ?: 0.0 }
        val mlConfidence = probByRisk[predictedRisk] ?: 0.0

        return MlPredictionResult(
            mlPrediction       = predictedRisk,
            mlConfidence       = mlConfidence,
            mlProbabilities    = mlProbabilities,
            aqiDerivedRisk     = aqiDerivedRisk,
            rawFeatures        = rawFeatures,
            scaledFeatures     = scaled.map { it.toDouble() }
        )
    }

    /**
     * Extract the 7 inference features from a SensorReading.
     * AQI is deliberately excluded.
     */
    private fun extractFeatures(r: SensorReading): DoubleArray = doubleArrayOf(
        r.temperature,  // [0] Temperature (°C)
        r.humidity,     // [1] Humidity (%)
        r.pressure,     // [2] Pressure (hPa)
        r.gasKOhms,     // [3] Gas (kOhms)      — NOT AQI
        r.altitude,     // [4] Altitude (m)
        r.dewPoint,     // [5] Dew Point (°C)
        r.heatIndex     // [6] Heat Index (°C)
        // r.aqi is intentionally omitted
    )

    /** Release ONNX Runtime resources. Call from ViewModel.onCleared(). */
    @Synchronized
    fun close() {
        try {
            session?.close()
            env?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing ONNX session", e)
        } finally {
            session = null
            env = null
        }
    }
}

/**
 * Result from the ML model, carrying both the ML prediction and the
 * AQI-derived classification for side-by-side display.
 *
 * These two values are independent and may differ:
 * - [aqiDerivedRisk]: deterministic threshold applied to the sensor's AQI output
 * - [mlPrediction]: probabilistic inference from 7 sensor features
 */
data class MlPredictionResult(
    /** Risk level predicted by the ML model from 7 sensor features (not AQI). */
    val mlPrediction: RiskLevel,

    /** Probability of the predicted class (0.0–1.0). */
    val mlConfidence: Double,

    /**
     * Probabilities for all 6 risk levels in canonical order:
     * GOOD, MODERATE, UNHEALTHY_SENSITIVE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS.
     * VERY_UNHEALTHY is always 0.0 (class not present in training data).
     */
    val mlProbabilities: List<Double>,

    /** Risk level derived from the sensor's AQI value via RiskLevel.fromAqi(). */
    val aqiDerivedRisk: RiskLevel,

    /** Raw 7-feature input array (before scaling). */
    val rawFeatures: DoubleArray,

    /** Scaled 7-feature input array (after StandardScaler). */
    val scaledFeatures: List<Double>
) {
    val mlConfidencePercent: Int get() = (mlConfidence * 100).toInt()

    /** True if the ML prediction and AQI-derived label agree. */
    val classificationsAgree: Boolean get() = mlPrediction == aqiDerivedRisk

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MlPredictionResult) return false
        return mlPrediction == other.mlPrediction &&
            aqiDerivedRisk == other.aqiDerivedRisk &&
            mlConfidence == other.mlConfidence
    }

    override fun hashCode(): Int = mlPrediction.hashCode() * 31 + aqiDerivedRisk.hashCode()
}
