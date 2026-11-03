package com.airguard.ai.ui.analysis

import android.graphics.Color
import android.os.Bundle
import android.view.*
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.airguard.ai.data.RiskLevel
import com.airguard.ai.databinding.FragmentAnalysisBinding
import com.airguard.ai.ml.AirQualityClassifier
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.ValueFormatter

class AnalysisFragment : Fragment() {

    private var _binding: FragmentAnalysisBinding? = null
    private val binding get() = _binding!!
    private val viewModel: AnalysisViewModel by viewModels()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentAnalysisBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupCharts()
        observeViewModel()
    }

    private fun observeViewModel() {
        viewModel.latestPrediction.observe(viewLifecycleOwner) { prediction ->
            prediction ?: return@observe
            val risk = prediction.riskLevel
            binding.tvRiskEmoji.text = risk.emoji
            binding.tvRiskName.text = risk.label
            binding.tvRiskDescription.text = risk.description
            binding.tvConfidence.text = "${prediction.confidencePercent}%"
            try { binding.tvRiskName.setTextColor(Color.parseColor(risk.colorHex)) } catch (_: Exception) {}

            // Feature contributions
            binding.featureContainer.removeAllViews()
            val topFeatures = prediction.topFeaturesByImportance()
            topFeatures.forEach { (name, importance) ->
                val row = layoutInflater.inflate(android.R.layout.simple_list_item_1, binding.featureContainer, false)
                // Build a simple row: label + bar
                val container = LinearLayout(requireContext()).apply {
                    orientation = LinearLayout.HORIZONTAL
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).also { it.bottomMargin = 6 }
                }
                val label = TextView(requireContext()).apply {
                    text = name
                    textSize = 12f
                    setTextColor(Color.parseColor("#666680"))
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                }
                val pct = TextView(requireContext()).apply {
                    text = "${(importance * 100).toInt()}%"
                    textSize = 12f
                    setTextColor(Color.parseColor("#1565C0"))
                    textAlignment = View.TEXT_ALIGNMENT_TEXT_END
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                }
                container.addView(label)
                container.addView(pct)
                binding.featureContainer.addView(container)
            }
        }

        viewModel.aqiDistribution.observe(viewLifecycleOwner) { dist ->
            dist ?: return@observe
            val total = dist.values.sum().toDouble()
            if (total == 0.0) return@observe

            binding.tvStatTotal.text = "%,d".format(total.toInt())
            binding.tvStatGood.text = "%.0f%%".format((dist[RiskLevel.GOOD] ?: 0) / total * 100)
            binding.tvStatModerate.text = "%.0f%%".format(((dist[RiskLevel.MODERATE] ?: 0) + (dist[RiskLevel.UNHEALTHY_SENSITIVE] ?: 0)) / total * 100)
            binding.tvStatUnhealthy.text = "%.0f%%".format(((dist[RiskLevel.UNHEALTHY] ?: 0) + (dist[RiskLevel.VERY_UNHEALTHY] ?: 0) + (dist[RiskLevel.HAZARDOUS] ?: 0)) / total * 100)

            updatePieChart(dist, total)
        }

        viewModel.recentTrendData.observe(viewLifecycleOwner) { data ->
            data ?: return@observe
            updateCombinedChart(data)
        }

        viewModel.riskHistoryData.observe(viewLifecycleOwner) { data ->
            data ?: return@observe
            updateRiskHistoryChart(data)
        }
    }

    private fun setupCharts() {
        // Pie chart
        binding.chartAqiDistribution.apply {
            description.isEnabled = false
            isDrawHoleEnabled = true
            holeRadius = 50f
            setHoleColor(Color.WHITE)
            setDrawEntryLabels(true)
            setEntryLabelTextSize(10f)
            legend.isEnabled = true
            setNoDataText("No data yet")
        }

        // Bar chart (risk history)
        binding.chartRiskHistory.apply {
            description.isEnabled = false
            legend.isEnabled = false
            setNoDataText("No data yet")
            xAxis.position = XAxis.XAxisPosition.BOTTOM
            xAxis.setDrawGridLines(false)
            axisRight.isEnabled = false
            axisLeft.granularity = 1f
        }

        // Combined chart
        binding.chartCombined.apply {
            description.isEnabled = false
            setNoDataText("No data yet")
            xAxis.position = XAxis.XAxisPosition.BOTTOM
            xAxis.setDrawGridLines(false)
            axisRight.isEnabled = false
        }
    }

    private fun updatePieChart(dist: Map<RiskLevel, Int>, total: Double) {
        val entries = listOf(
            PieEntry((dist[RiskLevel.GOOD] ?: 0).toFloat(), "Good"),
            PieEntry((dist[RiskLevel.MODERATE] ?: 0).toFloat(), "Moderate"),
            PieEntry((dist[RiskLevel.UNHEALTHY_SENSITIVE] ?: 0).toFloat(), "Sensitive"),
            PieEntry((dist[RiskLevel.UNHEALTHY] ?: 0).toFloat(), "Unhealthy"),
            PieEntry((dist[RiskLevel.VERY_UNHEALTHY] ?: 0).toFloat(), "Very Unhealthy"),
            PieEntry((dist[RiskLevel.HAZARDOUS] ?: 0).toFloat(), "Hazardous"),
        ).filter { it.value > 0 }

        val colors = listOf(
            Color.parseColor("#4CAF50"),
            Color.parseColor("#FFC107"),
            Color.parseColor("#FF9800"),
            Color.parseColor("#F44336"),
            Color.parseColor("#9C27B0"),
            Color.parseColor("#B71C1C")
        )

        val dataSet = PieDataSet(entries, "").apply {
            this.colors = colors.take(entries.size)
            valueTextSize = 10f
            valueTextColor = Color.WHITE
            sliceSpace = 2f
        }

        binding.chartAqiDistribution.data = PieData(dataSet)
        binding.chartAqiDistribution.invalidate()
    }

    private fun updateRiskHistoryChart(data: List<Pair<String, Int>>) {
        if (data.isEmpty()) return
        val entries = data.mapIndexed { i, (_, count) -> BarEntry(i.toFloat(), count.toFloat()) }
        val labels = data.map { it.first }

        val dataSet = BarDataSet(entries, "Risk Events").apply {
            color = Color.parseColor("#1565C0")
            valueTextSize = 10f
        }

        binding.chartRiskHistory.xAxis.valueFormatter = object : ValueFormatter() {
            override fun getFormattedValue(value: Float): String =
                labels.getOrElse(value.toInt()) { "" }
        }
        binding.chartRiskHistory.data = BarData(dataSet)
        binding.chartRiskHistory.invalidate()
    }

    private fun updateCombinedChart(data: List<Triple<String, Double, Double>>) {
        if (data.isEmpty()) return
        val tempEntries = data.mapIndexed { i, (_, t, _) -> Entry(i.toFloat(), t.toFloat()) }
        val humidEntries = data.mapIndexed { i, (_, _, h) -> Entry(i.toFloat(), h.toFloat()) }
        val labels = data.map { it.first }

        val tempSet = LineDataSet(tempEntries, "Temp (°C)").apply {
            color = Color.parseColor("#F44336")
            setCircleColor(Color.parseColor("#F44336"))
            circleRadius = 2f
            lineWidth = 2f
            setDrawValues(false)
            mode = LineDataSet.Mode.CUBIC_BEZIER
        }
        val humidSet = LineDataSet(humidEntries, "Humidity (%)").apply {
            color = Color.parseColor("#448AFF")
            setCircleColor(Color.parseColor("#448AFF"))
            circleRadius = 2f
            lineWidth = 2f
            setDrawValues(false)
            mode = LineDataSet.Mode.CUBIC_BEZIER
        }

        binding.chartCombined.xAxis.valueFormatter = object : ValueFormatter() {
            override fun getFormattedValue(value: Float): String =
                labels.getOrElse(value.toInt()) { "" }
        }

        val combinedData = CombinedData()
        combinedData.setData(LineData(tempSet, humidSet))
        binding.chartCombined.data = combinedData
        binding.chartCombined.invalidate()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
