package com.airguard.ai.ui.settings

import android.app.Application
import androidx.lifecycle.*
import com.airguard.ai.data.SensorRepository
import com.airguard.ai.data.SyncResult
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

class SettingsViewModel(app: Application) : AndroidViewModel(app) {

    private val repo = SensorRepository(app)

    private val _lastSync = MutableLiveData<Long>()
    val lastSync: LiveData<Long> = _lastSync

    private val _cacheSize = MutableLiveData<Int>()
    val cacheSize: LiveData<Int> = _cacheSize

    private val _syncMessage = MutableLiveData<String?>()
    val syncMessage: LiveData<String?> = _syncMessage

    init {
        refresh()
    }

    private fun refresh() {
        _lastSync.value = repo.getLastSyncTime()
        viewModelScope.launch {
            _cacheSize.value = repo.getCount()
        }
    }

    fun syncNow() {
        viewModelScope.launch {
            _syncMessage.value = "Syncing..."
            when (val result = repo.syncFromSheets()) {
                is SyncResult.Success -> {
                    _syncMessage.value = "✅ Synced ${result.totalRows} total readings"
                    refresh()
                }
                is SyncResult.Failure -> {
                    _syncMessage.value = "❌ Sync failed: ${result.reason}"
                }
            }
        }
    }

    fun clearCache() {
        viewModelScope.launch {
            repo.clearAll()
            _syncMessage.value = "Cache cleared"
            refresh()
        }
    }
}
