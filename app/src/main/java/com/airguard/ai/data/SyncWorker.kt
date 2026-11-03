package com.airguard.ai.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SyncWorker(private val context: Context) {
    suspend fun doWork(): Boolean {
        val repo = SensorRepository(context)
        return when (repo.syncFromSheets()) {
            is SyncResult.Success -> true
            is SyncResult.Failure -> false
        }
    }
}
