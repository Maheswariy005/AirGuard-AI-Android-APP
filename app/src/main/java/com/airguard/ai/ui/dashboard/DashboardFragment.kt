package com.airguard.ai.ui.dashboard

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.airguard.ai.data.RiskLevel
import com.airguard.ai.data.SensorReading
import com.airguard.ai.databinding.FragmentDashboardBinding
import com.airguard.ai.ml.PredictionResult
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.ValueFormatter
import java.text.SimpleDateFormat
import java.util.*

class DashboardFragment : Fragment() {

    private var _binding: FragmentDashboardBinding? = null
    private val binding get() = _binding!!
    private val viewModel: DashboardViewModel by viewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDashboardBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupChart()
        observeViewModel()
        setupClickListeners()
        updateConnectivityStatus()

        // Show stale banner if needed
        if (viewModel.isDataStale) {
            binding.bannerOutdated.visibility = View.VISIBLE
        }
    }

    private fun setupClickListeners() {
        binding.btnSync.setOnClickListener { viewModel.triggerSync() }
        binding.btnSyncBanner.setOnClickListener { viewModel.triggerSync() }
    }

    private fun observeViewModel() {
        viewModel.latestReading.observe(viewLifecycleOwner) { reading ->
            if (reading != null) {
                updateSensorCards(reading)
                binding.bannerOutdated.visibility =
                    if (viewModel.isDataStale) View.VISIBLE else View.GONE
            }
        }

        viewModel.recentReadings.observe(viewLifecycleOwner) { readings ->
            updateChart(readings)
        }

        viewModel.prediction.observe(viewLifecycleOwner) { prediction ->
            if (prediction != null) {
                updatePredictionCard(prediction)
            }
        }

        viewModel.syncState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is SyncState.Syncing -> {
                    binding.btnSync.isEnabled = false
                    binding.btnSync.text = "⏳  Syncing..."
                }
                is SyncState.Success -> {
                    binding.btnSync.isEnabled = true
                    binding.btnSync.text = "🔄  Sync Latest Data"
                    binding.tvLastSync.text = "Synced: ${viewModel.getLastSyncFormatted()}"
                    binding.bannerOutdated.visibility = View.GONE
                    updateConnectivityStatus()
                    Toast.makeText(
                        requireContext(),
                        "✅ Synced ${state.totalRows} total readings",
                        Toast.LENGTH_SHORT
                    ).show()
                }
                is SyncState.Error -> {
                    binding.btnSync.isEnabled = true
                    binding.btnSync.text = "🔄  Sync Latest Data"
                    Toast.makeText(
                        requireContext(),
                        "❌ Sync failed: ${state.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
                else -> {}
            }
        }
    }

    private fun updateSensorCards(r: SensorReading) {
        binding.tvTempValue.text = "%.1f°C".format(r.temperature)
        binding.tvHumidityValue.text = "%.1f%%".format(r.humidity)
        binding.tvPressureValue.text = "%.1f".format(r.pressure)
        binding.tvGasValue.text = "%.1f".format(r.gasKOhms)
        binding.tvDewpointValue.text = "%.1f°C".format(r.dewPoint)
        binding.tvHeatindexValue.text = "%.1f°C".format(r.heatIndex)
        binding.tvAqiValue.text = r.aqi.toString()
        binding.tvReadingsTimestamp.text = r.time
        binding.tvLastSync.text = "Synced: ${viewModel.getLastSyncFormatted()}"

        // Update AQI risk badge
        val risk = r.riskLevel
        binding.tvRiskLevel.text = "Risk: ${risk.label}"
        try {
            val color = Color.parseColor(risk.colorHex)
            binding.riskIndicatorDot.setBackgroundColor(color)
        } catch (_: Exception) {}
    }

    private fun updatePredictionCard(prediction: PredictionResult) {
        val risk = prediction.riskLevel
        binding.tvPredictionValue.text = risk.label
        binding.tvPredictionConfidence.text = "Confidence: ${prediction.confidencePercent}%"
        binding.tvPredictionIcon.text = risk.emoji
        binding.tvPredictionExplanation.text = risk.advice
        binding.tvModelTag.text = "On-device ML • Random Forest • ${prediction.confidencePercent}% confidence"

        try {
            binding.tvPredictionValue.setTextColor(Color.parseColor(risk.colorHex))
        } catch (_: Exception) {}
    }

    private fun updateConnectivityStatus() {
        if (viewModel.isOnline) {
            binding.tvConnectivity.text = "Connected"
            binding.statusDot.setBackgroundColor(Color.parseColor("#4CAF50"))
        } else {
            binding.tvConnectivity.text = "Offline"
            binding.statusDot.setBackgroundColor(Color.parseColor("#F44336"))
        }
    }

    private fun setupChart() {
        val chart = binding.chartTrend
        chart.description.isEnabled = false
        chart.legend.isEnabled = true
        chart.setTouchEnabled(true)
        chart.setScaleEnabled(true)
        chart.setDrawGridBackground(false)
        chart.setNoDataText("Sync data to see the trend")

        chart.xAxis.apply {
            position = XAxis.XAxisPosition.BOTTOM
            setDrawGridLines(false)
            labelRotationAngle = -45f
            textSize = 9f
        }
        chart.axisRight.isEnabled = false
        chart.axisLeft.apply {
            textSize = 10f
            setDrawGridLines(true)
            gridColor = Color.parseColor("#E0E0E0")
        }
    }

    private fun updateChart(readings: List<SensorReading>) {
        if (readings.isEmpty()) return

        val chart = binding.chartTrend
        val sorted = readings.sortedWith(compareBy({ it.date }, { it.time }))
            .takeLast(24) // Last 24 points for the chart

        val aqiEntries = sorted.mapIndexed { i, r ->
            Entry(i.toFloat(), r.aqi.toFloat())
        }

        val timeLabels = sorted.map { it.time.substring(0, 5) } // HH:mm

        val aqiSet = LineDataSet(aqiEntries, "AQI").apply {
            color = Color.parseColor("#FF6D00")
            setCircleColor(Color.parseColor("#FF6D00"))
            circleRadius = 3f
            lineWidth = 2f
            setDrawValues(false)
            mode = LineDataSet.Mode.CUBIC_BEZIER
        }

        chart.xAxis.valueFormatter = object : ValueFormatter() {
            override fun getFormattedValue(value: Float): String {
                val i = value.toInt()
                return if (i >= 0 && i < timeLabels.size) timeLabels[i] else ""
            }
        }

        chart.data = LineData(aqiSet)
        chart.invalidate()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
