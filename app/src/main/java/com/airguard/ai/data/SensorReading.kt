package com.airguard.ai.data

/**
 * Represents a single sensor reading row from the BME680/AQI sensor.
 * Fields match the spreadsheet columns exactly.
 */
data class SensorReading(
    /** Date string YYYY-MM-DD */
    val date: String,

    /** Time string HH:mm:ss */
    val time: String,

    /** Temperature in °C */
    val temperature: Double,

    /** Relative humidity in % */
    val humidity: Double,

    /** Atmospheric pressure in hPa */
    val pressure: Double,

    /** Air Quality Index (sensor-derived, scale 10–100) */
    val aqi: Int,

    /** Gas resistance in kΩ – higher = cleaner air */
    val gasKOhms: Double,

    /** Altitude in metres above sea level */
    val altitude: Double,

    /** Dew point temperature in °C */
    val dewPoint: Double,

    /** Heat index in °C */
    val heatIndex: Double,

    /** Unix epoch milliseconds when this row was stored locally */
    val storedAt: Long = System.currentTimeMillis()
) {
    /** Combined ISO-like timestamp for display */
    val dateTime: String get() = "$date $time"

    /** Risk label derived from AQI */
    val riskLevel: RiskLevel get() = RiskLevel.fromAqi(aqi)
}

/**
 * AQI-based risk classification following US EPA breakpoints adapted to
 * the sensor's 10–100 integer output scale.
 *
 * The sensor outputs AQI in multiples of 20 (10, 20, 40, 60, 80, 100).
 * We map these to EPA categories proportionally.
 *
 * NOTE: The AQI values are sensor-reported integers, NOT validated
 * against a calibrated reference. These thresholds are scientifically
 * defensible approximations; cross-validation with certified equipment
 * is required for production health applications.
 */
enum class RiskLevel(
    val label: String,
    val emoji: String,
    val colorHex: String,
    val description: String,
    val advice: String,
    val numericId: Int
) {
    GOOD(
        label = "Good",
        emoji = "✅",
        colorHex = "#4CAF50",
        description = "Air quality is satisfactory and poses little or no risk.",
        advice = "Air quality is good. Enjoy indoor activities.",
        numericId = 0
    ),
    MODERATE(
        label = "Moderate",
        emoji = "🟡",
        colorHex = "#FFC107",
        description = "Air quality is acceptable; minor concern for sensitive individuals.",
        advice = "Unusually sensitive people should consider reducing prolonged exertion.",
        numericId = 1
    ),
    UNHEALTHY_SENSITIVE(
        label = "Unhealthy (Sensitive)",
        emoji = "🟠",
        colorHex = "#FF9800",
        description = "Members of sensitive groups may experience health effects.",
        advice = "People with respiratory conditions should limit prolonged exertion.",
        numericId = 2
    ),
    UNHEALTHY(
        label = "Unhealthy",
        emoji = "🔴",
        colorHex = "#F44336",
        description = "Everyone may begin to experience health effects.",
        advice = "Everyone should reduce prolonged or heavy exertion. Ventilate the area.",
        numericId = 3
    ),
    VERY_UNHEALTHY(
        label = "Very Unhealthy",
        emoji = "🟣",
        colorHex = "#9C27B0",
        description = "Health alert: everyone may experience serious health effects.",
        advice = "Avoid all physical activity indoors. Improve ventilation immediately.",
        numericId = 4
    ),
    HAZARDOUS(
        label = "Hazardous",
        emoji = "⛔",
        colorHex = "#B71C1C",
        description = "Health emergency: everyone is affected.",
        advice = "Evacuate or seal the area. Seek fresh air immediately.",
        numericId = 5
    );

    companion object {
        /**
         * Map sensor AQI (10–100) to risk level.
         *
         * Sensor AQI 10–20  → Good          (EPA 0–50)
         * Sensor AQI 40     → Moderate      (EPA 51–100)
         * Sensor AQI 60     → Unhealthy-Sensitive (EPA 101–150)
         * Sensor AQI 80     → Unhealthy     (EPA 151–200)
         * Sensor AQI 100    → Very Unhealthy/Hazardous (EPA 201+)
         */
        fun fromAqi(aqi: Int): RiskLevel = when {
            aqi <= 20  -> GOOD
            aqi <= 40  -> MODERATE
            aqi <= 60  -> UNHEALTHY_SENSITIVE
            aqi <= 80  -> UNHEALTHY
            aqi <= 90  -> VERY_UNHEALTHY
            else       -> HAZARDOUS
        }
    }
}
