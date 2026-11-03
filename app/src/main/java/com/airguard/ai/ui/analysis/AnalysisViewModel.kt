package com.airguard.ai.ui.analysis

import android.app.Application
import androidx.lifecycle.*
import com.airguard.ai.data.RiskLevel
import com.airguard.ai.data.SensorReading
import com.airguard.ai.data.SensorRepository
import com.airguard.ai.ml.AirQualityClassifier
import com.airguard.ai.ml.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AnalysisViewModel(app: Application) : AndroidViewModel(app) {

    private val repo = SensorRepository(app)
    private val classifier = AirQualityClassifier()

    private val _latestPrediction = MutableLiveData<PredictionResult?>()
    val latestPrediction: LiveData<PredictionResult?> = _latestPrediction

    private val _aqiDistribution = MutableLiveData<Map<RiskLevel, Int>?>()
    val aqiDistribution: LiveData<Map<RiskLevel, Int>?> = _aqiDistribution

    private val _recentTrendData = MutableLiveData<List<Triple<String, Double, Double>>?>()
    val recentTrendData: LiveData<List<Triple<String, Double, Double>>?> = _recentTrendData

    private val _riskHistoryData = MutableLiveData<List<Pair<String, Int>>?>()
    val riskHistoryData: LiveData<List<Pair<String, Int>>?> = _riskHistoryData

    init {
        loadAnalysis()
    }

    private fun loadAnalysis() {
        viewModelScope.launch {
            val readings = repo.getLatestReadings(500)

            if (readings.isEmpty()) return@launch

            // Latest prediction
            val latest = readings.first()
            _latestPrediction.value = withContext(Dispatchers.Default) {
                classifier.predict(latest)
            }

            // AQI distribution
            val dist = withContext(Dispatchers.Default) {
                val map = mutableMapOf<RiskLevel, Int>()
                RiskLevel.entries.forEach { map[it] = 0 }
                readings.forEach { r ->
                    val risk = r.riskLevel
                    map[risk] = (map[risk] ?: 0) + 1
                }
                map
            }
            _aqiDistribution.value = dist

            // Trend data (last 30 points)
            val trendReadings = readings.reversed().takeLast(30)
            _recentTrendData.value = trendReadings.map { r ->
                Triple(r.time.take(5), r.temperature, r.humidity)
            }

            // Risk history - group by date, count unhealthy+
            val riskHistory = withContext(Dispatchers.Default) {
                readings.groupBy { it.date }
                    .entries
                    .sortedBy { it.key }
                    .takeLast(14)
                    .map { (date, dayReadings) ->
                        val unhealthy = dayReadings.count { it.aqi > 60 }
                        val shortDate = date.substring(5) // MM-DD
                        Pair(shortDate, unhealthy)
                    }
            }
            _riskHistoryData.value = riskHistory
        }
    }

    fun refresh() = loadAnalysis()
}
