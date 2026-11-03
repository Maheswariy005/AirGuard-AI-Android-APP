package com.airguard.ai.data

import android.content.Context
import android.content.SharedPreferences
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

/**
 * Single source of truth for sensor data.
 * Persists to JSON file on internal storage — no annotation processor needed.
 */
class SensorRepository(private val context: Context) {

    private val gson = Gson()
    private val prefs: SharedPreferences =
        context.getSharedPreferences("airguard_prefs", Context.MODE_PRIVATE)

    private val dataFile = File(context.filesDir, "sensor_readings.json")

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    companion object {
        private const val TAG = "SensorRepository"
        private const val SHEET_CSV_URL =
            "https://docs.google.com/spreadsheets/d/13BOzm1LVNGvoJ2wspWgEztHdRzOS6QmOCoNKwiSwB4E/export?format=csv&gid=0"
        private const val PREF_LAST_SYNC = "last_sync_time"
        private const val MAX_STORED = 2000  // keep latest 2000 readings
    }

    // ────────────────── In-memory cache ──────────────────
    private var _cache: MutableList<SensorReading>? = null

    private suspend fun loadCache(): MutableList<SensorReading> = withContext(Dispatchers.IO) {
        _cache ?: run {
            val list = if (dataFile.exists()) {
                try {
                    val json = dataFile.readText()
                    val type = object : TypeToken<List<SensorReading>>() {}.type
                    gson.fromJson<List<SensorReading>>(json, type)?.toMutableList()
                        ?: mutableListOf()
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load cache", e)
                    mutableListOf()
                }
            } else {
                mutableListOf()
            }
            _cache = list
            list
        }
    }

    private suspend fun saveCache(list: List<SensorReading>) = withContext(Dispatchers.IO) {
        try {
            val json = gson.toJson(list)
            dataFile.writeText(json)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save cache", e)
        }
    }

    // ────────────────── Sync ──────────────────

    suspend fun syncFromSheets(): SyncResult = withContext(Dispatchers.IO) {
        if (!isNetworkAvailable()) {
            return@withContext SyncResult.Failure("No network connection")
        }
        try {
            Log.i(TAG, "Starting sync from Google Sheets")
            val request = Request.Builder().url(SHEET_CSV_URL).build()
            val response = httpClient.newCall(request).execute()
            if (!response.isSuccessful) {
                return@withContext SyncResult.Failure("HTTP ${response.code}")
            }
            val body = response.body?.string()
                ?: return@withContext SyncResult.Failure("Empty response")

            val readings = parseCsv(body)
            Log.i(TAG, "Parsed ${readings.size} readings")
            if (readings.isEmpty()) {
                return@withContext SyncResult.Failure("No valid readings parsed")
            }

            // Keep latest MAX_STORED entries
            val toStore = readings.sortedWith(compareBy({ it.date }, { it.time }))
                .takeLast(MAX_STORED)

            _cache = toStore.toMutableList()
            saveCache(toStore)

            prefs.edit().putLong(PREF_LAST_SYNC, System.currentTimeMillis()).apply()
            SyncResult.Success(readings.size, toStore.size)

        } catch (e: Exception) {
            Log.e(TAG, "Sync failed", e)
            SyncResult.Failure(e.message ?: "Unknown error")
        }
    }

    // ────────────────── CSV Parsing ──────────────────

    fun parseCsv(csv: String): List<SensorReading> {
        val lines = csv.lines()
        val readings = mutableListOf<SensorReading>()
        val headerIndex = lines.indexOfFirst { it.contains("Date") && it.contains("Time") }
        if (headerIndex < 0) return emptyList()

        for (i in (headerIndex + 1)..lines.lastIndex) {
            val line = lines[i].trim()
            if (line.isEmpty()) continue
            try {
                val cols = line.split(",")
                if (cols.size < 10) continue
                val date = cols[0].trim()
                if (!date.matches(Regex("""\d{4}-\d{2}-\d{2}"""))) continue
                val time = cols[1].trim()
                val temp = cols[2].trim().toDoubleOrNull() ?: continue
                val humidity = cols[3].trim().toDoubleOrNull() ?: continue
                val pressure = cols[4].trim().toDoubleOrNull() ?: continue
                val aqi = cols[5].trim().toIntOrNull() ?: continue
                val gas = cols[6].trim().toDoubleOrNull() ?: continue
                val altitude = cols[7].trim().toDoubleOrNull() ?: continue
                val dewPoint = cols[8].trim().toDoubleOrNull() ?: continue
                val heatIndex = cols[9].trim().toDoubleOrNull() ?: continue
                if (temp < -40 || temp > 85 || humidity < 0 || humidity > 100) continue
                readings.add(SensorReading(date, time, temp, humidity, pressure, aqi, gas, altitude, dewPoint, heatIndex))
            } catch (_: Exception) {}
        }
        return readings
    }

    // ────────────────── Queries ──────────────────

    suspend fun getMostRecent(): SensorReading? {
        val cache = loadCache()
        return cache.maxByOrNull { "${it.date} ${it.time}" }
    }

    suspend fun getLatestReadings(limit: Int = 200): List<SensorReading> {
        val cache = loadCache()
        return cache.sortedWith(compareByDescending<SensorReading> { it.date }
            .thenByDescending { it.time }).take(limit)
    }

    suspend fun getAll(): List<SensorReading> = loadCache()

    suspend fun getCount(): Int = loadCache().size

    suspend fun getLastNDays(days: Int): List<SensorReading> {
        val cal = Calendar.getInstance()
        cal.add(Calendar.DAY_OF_YEAR, -days)
        val fmt = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        val fromDate = fmt.format(cal.time)
        val cache = loadCache()
        return cache.filter { it.date >= fromDate }
    }

    suspend fun getFromDate(fromDate: String): List<SensorReading> {
        val cache = loadCache()
        return cache.filter { it.date >= fromDate }
            .sortedWith(compareByDescending<SensorReading> { it.date }.thenByDescending { it.time })
    }

    suspend fun clearAll() {
        _cache = mutableListOf()
        dataFile.delete()
        prefs.edit().remove(PREF_LAST_SYNC).apply()
    }

    // ────────────────── Prefs ──────────────────

    fun getLastSyncTime(): Long = prefs.getLong(PREF_LAST_SYNC, 0L)

    fun isDataStale(thresholdMs: Long = 3_600_000L): Boolean {
        val last = getLastSyncTime()
        return last == 0L || (System.currentTimeMillis() - last) > thresholdMs
    }

    fun isNetworkAvailable(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(network) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }
}

sealed class SyncResult {
    data class Success(val newRows: Int, val totalRows: Int) : SyncResult()
    data class Failure(val reason: String) : SyncResult()
}
