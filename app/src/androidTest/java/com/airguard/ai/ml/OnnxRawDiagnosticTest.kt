package com.airguard.ai.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.airguard.ai.data.RiskLevel
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.FloatBuffer

/**
 * READ-ONLY DIAGNOSTIC TEST — Do not ship to production.
 *
 * Purpose: inspect the raw ONNX Runtime Android output tensor for the four
 * probability comparisons that exceeded 0.001 during instrumented testing,
 * and determine whether the discrepancy originates inside ORT or in Kotlin
 * extraction.
 *
 * This test opens the ONNX model directly (bypassing OnnxClassifier) so we
 * can capture the raw OrtValue output before any Kotlin processing.
 *
 * Failing vectors (from previous instrumented run):
 *   Vector index=1 (MODERATE): android GOOD=0.030823, python=0.035823, diff=5.00e-3
 *   Vector index=1 (MODERATE): android MOD =0.783123, python=0.777429, diff=5.69e-3
 *   Vector index=5 (MODERATE): android UNH_S=0.636034, python=0.641034, diff=5.00e-3
 *   Vector index=5 (MODERATE): android UNH  =0.161800, python=0.156800, diff=5.00e-3
 *
 * No application source is modified. This file is in androidTest only.
 */
@RunWith(AndroidJUnit4::class)
class OnnxRawDiagnosticTest {

    companion object {
        private const val TAG = "OnnxDiag"
        private const val MODEL_ASSET = "airguard_model.onnx"

        // LabelEncoder alphabetical order: 0=GOOD,1=HAZARDOUS,2=MODERATE,3=UNHEALTHY,4=UNH_S
        val CLASS_NAMES = arrayOf("GOOD", "HAZARDOUS", "MODERATE", "UNHEALTHY", "UNHEALTHY_SENSITIVE")

        // Python ORT-17 reference values for the failing vectors (from check_vec1.py)
        // Vector index=1 (MODERATE, raw features shown in test_vectors.json):
        val PYTHON_ORT17_V1 = mapOf(
            "GOOD" to 0.035823f, "HAZARDOUS" to 0.0f, "MODERATE" to 0.777429f,
            "UNHEALTHY" to 0.038540f, "UNHEALTHY_SENSITIVE" to 0.148208f
        )
        // Vector index=5 (true=MODERATE, actual entry in test_vectors.json array position 13):
        // We'll identify it by scanning for index==5

        private val FEATURE_KEYS = listOf(
            "Temperature\u00b0C", // matched below with endsWith
            "Humidity%",
            "PressurehPa",
            "GaskOhms",
            "Altitudem",
            "DewPoint\u00b0C",
            "HeatIndex\u00b0C"
        )
        private val FEAT_ORDERED = listOf(
            "Temperature(\u00b0C)",
            "Humidity(%)",
            "Pressure(hPa)",
            "Gas (kOhms)",
            "Altitude(m)",
            "DewPoint(\u00b0C)",
            "HeatIndex(\u00b0C)"
        )
    }

    // JSON data classes (same as OnnxInferenceTest)
    data class TV(@SerializedName("n_vectors") val n: Int,
                  @SerializedName("vectors") val vectors: List<Vec>)
    data class Vec(@SerializedName("index") val index: Int,
                   @SerializedName("true_class") val trueClass: String,
                   @SerializedName("predicted_class") val predictedClass: String,
                   @SerializedName("raw_features") val rawFeatures: Map<String, Double>,
                   @SerializedName("probabilities_by_encoded_int") val probs: Map<String, Double>)

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var testVectors: List<Vec>

    @Before
    fun setUp() {
        val appCtx  = InstrumentationRegistry.getInstrumentation().targetContext
        val testCtx = InstrumentationRegistry.getInstrumentation().context

        env = OrtEnvironment.getEnvironment()
        val bytes = appCtx.assets.open(MODEL_ASSET).readBytes()
        session = env.createSession(bytes, OrtSession.SessionOptions())

        val json = testCtx.assets.open("test_vectors.json").bufferedReader().use { it.readText() }
        testVectors = Gson().fromJson(json, TV::class.java).vectors
    }

    @After
    fun tearDown() {
        session.close()
        env.close()
    }

    // ── Helper: run a single sample directly through ORT, return raw results ──

    private data class RawOrtResult(
        val labelArray: LongArray,          // output "label"
        val probaArray: FloatArray,         // output "probabilities" - raw flat array
        val probaShape: LongArray,          // tensor shape
        val probaTypeName: String,          // element type
        val scaledInput: FloatArray         // what we sent in
    )

    private fun runRaw(rawFeatures: DoubleArray): RawOrtResult {
        // Apply StandardScaler constants (same as FeatureScaler.kt)
        val means  = doubleArrayOf(32.404256, 65.030945, 1002.640914, 253.995597, 88.799590, 25.388473, 39.360001)
        val scales = doubleArrayOf( 4.205421,  7.397750,    5.991025,  89.545656, 50.336149,  4.758210,  7.739075)
        val scaled = FloatArray(7) { i -> ((rawFeatures[i] - means[i]) / scales[i]).toFloat() }

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(scaled), longArrayOf(1L, 7L))
        val results = try {
            session.run(mapOf("float_input" to tensor))
        } finally {
            tensor.close()
        }
        try {
            @Suppress("UNCHECKED_CAST")
            val labels  = results[0].value as LongArray
            // Inspect the raw OrtValue for probabilities before any casting
            val probaOrtValue = results[1]
            val typeName = probaOrtValue.type.toString()

            // Try direct cast to Array<FloatArray> (2D)
            val rawValue = probaOrtValue.value
            val probaFlat: FloatArray = when (rawValue) {
                is Array<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    (rawValue as Array<FloatArray>)[0]
                }
                is FloatArray -> rawValue
                else -> floatArrayOf(-1f) // unexpected
            }
            // Get shape from the OnnxTensor if available
            val shape: LongArray = try {
                (probaOrtValue as? OnnxTensor)?.info?.shape ?: longArrayOf(-1)
            } catch (e: Exception) { longArrayOf(-1) }

            return RawOrtResult(labels, probaFlat, shape, typeName, scaled)
        } finally {
            results.close()
        }
    }

    // ══════════════════════════════════════════════════════════
    // DIAGNOSTIC 1: Raw tensor inspection for failing vectors
    // ══════════════════════════════════════════════════════════

    @Test
    fun diagnostic_rawTensorInspection() {
        val targetIndices = setOf(1, 5)
        val diagLog = StringBuilder()
        diagLog.appendLine("=== OnnxRawDiagnosticTest: Raw Tensor Inspection ===\n")

        for (vec in testVectors) {
            if (vec.index !in targetIndices) continue

            val raw = DoubleArray(FEAT_ORDERED.size) { i ->
                vec.rawFeatures[FEAT_ORDERED[i]] ?: 0.0
            }

            diagLog.appendLine("--- Vector index=${vec.index} true_class=${vec.trueClass} ---")
            diagLog.appendLine("Raw 7 features: ${raw.toList()}")

            val result = runRaw(raw)

            // 1. Shape and type of probability output
            diagLog.appendLine("ORT output[1] type    : ${result.probaTypeName}")
            diagLog.appendLine("ORT output[1] shape   : ${result.probaShape.toList()}")
            diagLog.appendLine("Scaled input sent     : ${result.scaledInput.toList()}")
            diagLog.appendLine("ORT predicted label   : ${result.labelArray[0]}")
            diagLog.appendLine("ORT predicted class   : ${CLASS_NAMES.getOrElse(result.labelArray[0].toInt()) { "UNKNOWN" }}")

            // 2. Raw probability array values
            diagLog.appendLine("Raw ORT probability array (${result.probaArray.size} elements):")
            var rawSum = 0.0
            for (i in result.probaArray.indices) {
                val cls = CLASS_NAMES.getOrElse(i) { "idx$i" }
                diagLog.appendLine("  [$i] $cls = ${result.probaArray[i]}  (double=${result.probaArray[i].toDouble()})")
                rawSum += result.probaArray[i]
            }
            diagLog.appendLine("Raw ORT probability SUM = $rawSum")

            // 3. Python ORT-17 reference from JSON
            diagLog.appendLine("Python ORT-17 reference (from test_vectors.json):")
            var jsonSum = 0.0
            val jsonProbs = vec.probs
            for (cls in CLASS_NAMES) {
                val p = jsonProbs[cls] ?: 0.0
                diagLog.appendLine("  $cls = $p")
                jsonSum += p
            }
            diagLog.appendLine("JSON probability SUM   = $jsonSum")

            // 4. Per-class diff
            diagLog.appendLine("Per-class absolute difference (Android raw - Python ORT-17):")
            for (i in result.probaArray.indices) {
                val cls = CLASS_NAMES.getOrElse(i) { "idx$i" }
                val android_p = result.probaArray[i].toDouble()
                val python_p  = jsonProbs[cls] ?: 0.0
                val diff = android_p - python_p
                diagLog.appendLine("  $cls: android=${result.probaArray[i]}  python=$python_p  diff=${"%.6e".format(diff)}")
            }

            // 5. Does the label (argmax) match the highest raw probability?
            val maxIdx = result.probaArray.indices.maxByOrNull { result.probaArray[it] } ?: -1
            val labelMatchesArgmax = result.labelArray[0].toInt() == maxIdx
            diagLog.appendLine("Label is argmax of raw probabilities: $labelMatchesArgmax  (label=${result.labelArray[0]}, argmax=$maxIdx)")

            diagLog.appendLine()
        }

        // Print to Android logcat AND to test output
        println(diagLog.toString())

        // Also assert for structured capture
        for (vec in testVectors) {
            if (vec.index !in targetIndices) continue
            val raw = DoubleArray(FEAT_ORDERED.size) { i -> vec.rawFeatures[FEAT_ORDERED[i]] ?: 0.0 }
            val result = runRaw(raw)

            // All 5 classes extracted (not 4 or 6)
            assertEquals("Vector ${vec.index}: probability array must have 5 elements", 5, result.probaArray.size)

            // Label matches argmax
            val maxIdx = result.probaArray.indices.maxByOrNull { result.probaArray[it] } ?: -1
            assertEquals("Vector ${vec.index}: ORT label must equal argmax of raw proba array",
                result.labelArray[0].toInt(), maxIdx)

            // No probability is NaN or infinite
            for (i in result.probaArray.indices) {
                assertTrue("Vector ${vec.index} class ${CLASS_NAMES[i]}: probability must be finite",
                    result.probaArray[i].isFinite())
                assertTrue("Vector ${vec.index} class ${CLASS_NAMES[i]}: probability must be >= 0",
                    result.probaArray[i] >= 0f)
            }
        }
    }

    // ══════════════════════════════════════════════════════════
    // DIAGNOSTIC 2: OnnxClassifier extraction fidelity
    // ══════════════════════════════════════════════════════════

    @Test
    fun diagnostic_classifierExtractionMatchesRaw() {
        // Verify OnnxClassifier.predictFromFeatures() returns the same probabilities
        // as the raw ORT run — confirming Kotlin extraction does not alter values.
        val appCtx = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = OnnxClassifier(appCtx)
        val diagLog = StringBuilder()
        diagLog.appendLine("=== Classifier extraction fidelity vs raw ORT ===\n")

        try {
            for (vec in testVectors) {
                if (vec.index !in setOf(1, 5)) continue

                val raw = DoubleArray(FEAT_ORDERED.size) { i -> vec.rawFeatures[FEAT_ORDERED[i]] ?: 0.0 }

                // Raw ORT (direct)
                val rawResult = runRaw(raw)

                // Via OnnxClassifier
                val classResult = classifier.predictFromFeatures(raw, RiskLevel.GOOD)

                diagLog.appendLine("Vector ${vec.index}:")
                diagLog.appendLine("  Raw ORT proba  : ${rawResult.probaArray.toList()}")
                diagLog.appendLine("  OnnxClassifier mlProbabilities (canonical order):")
                diagLog.appendLine("    ${classResult.mlProbabilities}")

                // Map raw ORT (alphabetical) to canonical for comparison
                // Raw:  [0=GOOD, 1=HAZARDOUS, 2=MODERATE, 3=UNHEALTHY, 4=UNH_S]
                // Canon:[0=GOOD, 1=MODERATE,  2=UNH_S,    3=UNHEALTHY, 4=VU,   5=HAZARDOUS]
                val rawToCanon = intArrayOf(0, 5, 1, 3, 2)  // raw[i] -> canon[rawToCanon[i]]
                val reconstructed = DoubleArray(6)
                for (i in 0..4) reconstructed[rawToCanon[i]] = rawResult.probaArray[i].toDouble()
                // VERY_UNHEALTHY stays 0

                diagLog.appendLine("  Reconstructed from raw (canonical): ${reconstructed.toList()}")

                var maxExtractionDiff = 0.0
                for (i in classResult.mlProbabilities.indices) {
                    val diff = kotlin.math.abs(classResult.mlProbabilities[i] - reconstructed[i])
                    if (diff > maxExtractionDiff) maxExtractionDiff = diff
                }
                diagLog.appendLine("  Max diff (OnnxClassifier vs raw ORT): ${"%.2e".format(maxExtractionDiff)}")
                diagLog.appendLine()

                // Assert: OnnxClassifier must faithfully transmit what ORT gives
                for (i in classResult.mlProbabilities.indices) {
                    assertEquals(
                        "Vector ${vec.index} canonical[$i]: OnnxClassifier must match raw ORT output",
                        reconstructed[i], classResult.mlProbabilities[i], 1e-7
                    )
                }
            }
        } finally {
            classifier.close()
        }
        println(diagLog.toString())
    }

    // ══════════════════════════════════════════════════════════
    // DIAGNOSTIC 3: Probability sum characterization
    // ══════════════════════════════════════════════════════════

    @Test
    fun diagnostic_probabilitySumAllVectors() {
        // For all 48 vectors, check whether probabilities sum to exactly 1.0.
        // The raw ORT output is used (not through OnnxClassifier).
        var maxDeviation = 0.0
        var nNotExactlyOne = 0
        val outliers = mutableListOf<String>()

        for (vec in testVectors) {
            val raw = DoubleArray(FEAT_ORDERED.size) { i -> vec.rawFeatures[FEAT_ORDERED[i]] ?: 0.0 }
            val result = runRaw(raw)
            val sum = result.probaArray.map { it.toDouble() }.sum()
            val deviation = kotlin.math.abs(sum - 1.0)
            if (deviation > maxDeviation) maxDeviation = deviation
            if (deviation > 1e-6) {
                nNotExactlyOne++
                outliers.add("Vector ${vec.index}: sum=${"%.8f".format(sum)}  dev=${"%.2e".format(deviation)}")
            }
        }

        println("=== Probability sum analysis (48 vectors) ===")
        println("Max deviation from 1.0: ${"%.6e".format(maxDeviation)}")
        println("Vectors with |sum-1| > 1e-6: $nNotExactlyOne / ${testVectors.size}")
        for (o in outliers) println("  $o")

        // Not asserting sum == 1.0 — this is diagnostic.
        // We only fail if the deviation is unreasonably large (> 0.01)
        assertTrue("Probability sums must stay within 1.0 ± 0.01", maxDeviation < 0.01)
    }
}
