package com.airguard.ai.data

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        // No-op: background sync handled manually
    }

    companion object {
        fun scheduleSyncWork(context: Context) {
            // No-op: WorkManager not used; sync is triggered manually
        }
    }
}
