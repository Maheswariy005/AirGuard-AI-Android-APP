package com.airguard.ai.ui.dashboard

import android.app.Application
import androidx.lifecycle.*
import com.airguard.ai.data.SensorReading
import com.airguard.ai.data.SensorRepository
import com.airguard.ai.data.SyncResult
import com.airguard.ai.ml.AirQualityClassifier
import com.airguard.ai.ml.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

class DashboardViewModel(app: Application) : AndroidViewModel(app) {

    private val repo = SensorRepository(app)
    private val classifier = AirQualityClassifier()

    private val _latestReading = MutableLiveData<SensorReading?>()
    val latestReading: LiveData<SensorReading?> = _latestReading

    private val _recentReadings = MutableLiveData<List<SensorReading>>(emptyList())
    val recentReadings: LiveData<List<SensorReading>> = _recentReadings

    private val _syncState = MutableLiveData<SyncState>(SyncState.Idle)
    val syncState: LiveData<SyncState> = _syncState

    private val _prediction = MutableLiveData<PredictionResult?>()
    val prediction: LiveData<PredictionResult?> = _prediction

    val isOnline: Boolean get() = repo.isNetworkAvailable()
    val isDataStale: Boolean get() = repo.isDataStale()

    init {
        loadData()
    }

    private fun loadData() {
        viewModelScope.launch {
            val recent = repo.getLatestReadings(50)
            _latestReading.value = recent.firstOrNull()
            _recentReadings.value = recent
            recent.firstOrNull()?.let { reading ->
                val result = withContext(Dispatchers.Default) { classifier.predict(reading) }
                _prediction.value = result
            }
        }
    }

    fun getLastSyncFormatted(): String {
        val ts = repo.getLastSyncTime()
        if (ts == 0L) return "Never"
        val fmt = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())
        return fmt.format(Date(ts))
    }

    fun triggerSync() {
        if (_syncState.value == SyncState.Syncing) return
        _syncState.value = SyncState.Syncing
        viewModelScope.launch {
            val result = repo.syncFromSheets()
            _syncState.value = when (result) {
                is SyncResult.Success -> SyncState.Success(result.newRows, result.totalRows)
                is SyncResult.Failure -> SyncState.Error(result.reason)
            }
            loadData()
        }
    }

    fun computePrediction() {
        viewModelScope.launch {
            val reading = repo.getMostRecent() ?: return@launch
            val result = withContext(Dispatchers.Default) { classifier.predict(reading) }
            _prediction.value = result
        }
    }
}

sealed class SyncState {
    object Idle : SyncState()
    object Syncing : SyncState()
    data class Success(val newRows: Int, val totalRows: Int) : SyncState()
    data class Error(val message: String) : SyncState()
}
