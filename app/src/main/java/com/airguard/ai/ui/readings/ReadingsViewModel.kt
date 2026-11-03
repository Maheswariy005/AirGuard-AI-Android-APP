package com.airguard.ai.ui.readings

import android.app.Application
import androidx.lifecycle.*
import com.airguard.ai.data.SensorReading
import com.airguard.ai.data.SensorRepository
import com.airguard.ai.data.SyncResult
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

enum class ReadingsFilter { ALL, TODAY, WEEK, MONTH }

class ReadingsViewModel(app: Application) : AndroidViewModel(app) {

    private val repo = SensorRepository(app)

    private val _filter = MutableLiveData(ReadingsFilter.ALL)
    private val _readings = MutableLiveData<List<SensorReading>>(emptyList())
    val readings: LiveData<List<SensorReading>> = _readings

    private val _readingCount = MutableLiveData<Int>(0)
    val readingCount: LiveData<Int> = _readingCount

    init {
        _filter.observeForever { applyFilter(it) }
    }

    private fun applyFilter(filter: ReadingsFilter) {
        viewModelScope.launch {
            val data = when (filter) {
                ReadingsFilter.ALL -> repo.getLatestReadings(500)
                ReadingsFilter.TODAY -> {
                    val today = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
                    repo.getFromDate(today)
                }
                ReadingsFilter.WEEK -> {
                    val cal = Calendar.getInstance()
                    cal.add(Calendar.DAY_OF_YEAR, -7)
                    val fromDate = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(cal.time)
                    repo.getFromDate(fromDate)
                }
                ReadingsFilter.MONTH -> {
                    val cal = Calendar.getInstance()
                    cal.add(Calendar.MONTH, -1)
                    val fromDate = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(cal.time)
                    repo.getFromDate(fromDate)
                }
            }
            _readings.value = data
            _readingCount.value = data.size
        }
    }

    fun setFilter(filter: ReadingsFilter) {
        _filter.value = filter
    }

    fun sync() {
        viewModelScope.launch {
            repo.syncFromSheets()
            applyFilter(_filter.value ?: ReadingsFilter.ALL)
        }
    }

    fun refreshCount() {
        viewModelScope.launch {
            _readingCount.value = repo.getCount()
        }
    }
}
