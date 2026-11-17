package com.airguard.ai.ml

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for FeatureScaler.
 *
 * Test vectors come from ml_pipeline/experiments/final_audit/test_vectors.json.
 * Expected scaled values were computed by the Python StandardScaler fitted on
 * training data only. These values confirm Kotlin matches Python arithmetic
 * before ONNX inference.
 *
 * To run: ./gradlew test
 */
class FeatureScalerTest {

    private val TOLERANCE = 1e-4   // float32 precision sufficient for ONNX

    /**
     * Verify scaler constants are present and feature count is correct.
     */
    @Test
    fun scalerHasCorrectFeatureCount() {
        assertEquals(7, FeatureScaler.FEATURE_COUNT)
        assertEquals(7, FeatureScaler.FEATURE_NAMES.size)
    }

    /**
     * AQI is NOT in the feature list.
     */
    @Test
    fun aqiNotInFeatureNames() {
        val names = FeatureScaler.FEATURE_NAMES.map { it.lowercase() }
        assertFalse("AQI must not be a feature", names.any { it.contains("aqi") })
    }

    /**
     * Scale a known UNHEALTHY sample from test_vectors.json (index 0).
     * Raw: Temperature=27.6, Humidity=72.1, Pressure=1003.2,
     *      Gas=211.0, Altitude=68.0, DewPoint=22.4, HeatIndex=31.6
     *
     * Expected values computed by Python StandardScaler (double precision,
     * then cast to float32 by FeatureScaler.scale()):
     *   Temperature : (27.6  - 32.404256) / 4.205421  = -1.142396
     *   Gas         : (211.0 - 253.995597) / 89.545656 = -0.480191
     *
     * The tolerance is 5e-4 to accommodate float32 rounding from the
     * double-precision Python intermediate result.
     */
    @Test
    fun scaleKnownUnhealthySample() {
        val raw = doubleArrayOf(27.6, 72.1, 1003.2, 211.0, 68.0, 22.4, 31.6)
        val scaled = FeatureScaler.scale(raw)

        assertEquals(7, scaled.size)
        // Temperature: (27.6 - 32.404256) / 4.205421 = -1.142396
        assertEquals(-1.1424f, scaled[0], 5e-4f)
        // Gas: (211.0 - 253.995597) / 89.545656 = -0.480191
        assertEquals(-0.4802f, scaled[3], 5e-4f)
    }

    /**
     * Scale a known HAZARDOUS sample from test_vectors.json.
     * Raw: Temperature=31.5, Humidity=52.8, Pressure=996.0,
     *      Gas=478.1, Altitude=144.6, DewPoint=22.0, HeatIndex=33.8
     * Expected Gas scaled: (478.1 - 253.995597) / 89.545656 = 2.5027
     */
    @Test
    fun scaleKnownHazardousSample() {
        val raw = doubleArrayOf(31.5, 52.8, 996.0, 478.1, 144.6, 22.0, 33.8)
        val scaled = FeatureScaler.scale(raw)
        // Gas (index 3): expect ~2.5027
        assertEquals(2.5027f, scaled[3], TOLERANCE.toFloat())
    }

    /**
     * Wrong size input throws.
     */
    @Test(expected = IllegalArgumentException::class)
    fun wrongSizeThrows() {
        FeatureScaler.scale(doubleArrayOf(1.0, 2.0, 3.0))
    }

    /**
     * NaN input throws.
     */
    @Test(expected = IllegalArgumentException::class)
    fun nanInputThrows() {
        FeatureScaler.scale(doubleArrayOf(30.0, 65.0, 1003.0, Double.NaN, 90.0, 25.0, 38.0))
    }

    /**
     * Infinite input throws.
     */
    @Test(expected = IllegalArgumentException::class)
    fun infiniteInputThrows() {
        FeatureScaler.scale(doubleArrayOf(30.0, 65.0, 1003.0, Double.POSITIVE_INFINITY, 90.0, 25.0, 38.0))
    }

    /**
     * Scale a "mean" input — all features equal their training mean.
     * Result should be all zeros.
     */
    @Test
    fun meanInputScalesToZero() {
        val meanInput = doubleArrayOf(32.404256, 65.030945, 1002.640914, 253.995597, 88.799590, 25.388473, 39.360001)
        val scaled = FeatureScaler.scale(meanInput)
        for (i in scaled.indices) {
            assertEquals("Feature $i should be ~0 at mean", 0.0f, scaled[i], TOLERANCE.toFloat())
        }
    }
}
