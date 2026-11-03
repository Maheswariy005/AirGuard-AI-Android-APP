package com.airguard.ai.ui.model

import android.graphics.Color
import android.os.Bundle
import android.view.*
import android.widget.LinearLayout
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.airguard.ai.data.RiskLevel
import com.airguard.ai.databinding.FragmentModelBinding
import com.airguard.ai.ml.AirQualityClassifier
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.ValueFormatter

class ModelFragment : Fragment() {

    private var _binding: FragmentModelBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentModelBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        buildClassificationTable()
        buildFeatureImportanceChart()
    }

    private fun buildClassificationTable() {
        val container = binding.classificationContainer
        container.removeAllViews()

        // Header
        val headerRow = makeRow(
            "AQI Range", "Risk Level", "Description",
            headerBg = "#1565C0", textColor = "#FFFFFF"
        )
        container.addView(headerRow)

        // Rows
        val rows = listOf(
            Triple("≤ 20", RiskLevel.GOOD, "#E8F5E9"),
            Triple("21–40", RiskLevel.MODERATE, "#FFFDE7"),
            Triple("41–60", RiskLevel.UNHEALTHY_SENSITIVE, "#FFF3E0"),
            Triple("61–80", RiskLevel.UNHEALTHY, "#FFEBEE"),
            Triple("81–90", RiskLevel.VERY_UNHEALTHY, "#F3E5F5"),
            Triple("91–100", RiskLevel.HAZARDOUS, "#FFCDD2")
        )

        rows.forEachIndexed { i, (range, risk, bg) ->
            val row = makeRow(
                range, "${risk.emoji} ${risk.label}", risk.description,
                rowBg = bg, textColor = "#1A1A2E",
                divider = i < rows.size - 1
            )
            container.addView(row)
        }

        // Note
        val note = TextView(requireContext()).apply {
            text = "* AQI values are sensor-reported integers (scale 10–100) mapped to US EPA breakpoints. Not calibrated against reference instruments."
            textSize = 11f
            setTextColor(Color.parseColor("#9999AA"))
            setPadding(0, 12, 0, 0)
        }
        container.addView(note)
    }

    private fun makeRow(
        col1: String, col2: String, col3: String,
        headerBg: String? = null, rowBg: String? = null,
        textColor: String = "#1A1A2E", divider: Boolean = false
    ): LinearLayout {
        val row = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            if (headerBg != null) setBackgroundColor(Color.parseColor(headerBg))
            else if (rowBg != null) setBackgroundColor(Color.parseColor(rowBg))
            setPadding(8, 10, 8, 10)
        }

        fun cell(text: String, weight: Float = 1f, bold: Boolean = false) = TextView(requireContext()).apply {
            this.text = text
            textSize = 11f
            setTextColor(Color.parseColor(textColor))
            if (bold) setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, weight)
        }

        row.addView(cell(col1, 0.8f, bold = headerBg != null))
        row.addView(cell(col2, 1.2f, bold = headerBg != null))
        row.addView(cell(col3, 2f))
        return row
    }

    private fun buildFeatureImportanceChart() {
        val chart = binding.chartFeatureImportance
        val names = AirQualityClassifier.FEATURE_NAMES
        val importance = AirQualityClassifier.FEATURE_IMPORTANCE

        // Sort descending
        val sorted = names.zip(importance.toList())
            .sortedByDescending { it.second }

        val entries = sorted.mapIndexed { i, (_, imp) ->
            BarEntry(i.toFloat(), (imp * 100).toFloat())
        }
        val labels = sorted.map { it.first }

        val dataSet = BarDataSet(entries, "Feature Importance (%)").apply {
            colors = listOf(
                Color.parseColor("#1565C0"),
                Color.parseColor("#00ACC1"),
                Color.parseColor("#F44336"),
                Color.parseColor("#FF9800"),
                Color.parseColor("#4CAF50"),
                Color.parseColor("#9C27B0"),
                Color.parseColor("#795548"),
                Color.parseColor("#607D8B")
            )
            valueTextSize = 10f
            valueFormatter = object : ValueFormatter() {
                override fun getBarLabel(barEntry: BarEntry?) =
                    "%.0f%%".format(barEntry?.y ?: 0f)
            }
        }

        chart.apply {
            description.isEnabled = false
            legend.isEnabled = false
            xAxis.apply {
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(false)
                valueFormatter = object : ValueFormatter() {
                    override fun getFormattedValue(value: Float): String =
                        labels.getOrElse(value.toInt()) { "" }
                }
                labelCount = labels.size
                textSize = 9f
            }
            axisRight.isEnabled = false
            axisLeft.apply {
                axisMinimum = 0f
                axisMaximum = 80f
            }
            data = BarData(dataSet).apply { barWidth = 0.6f }
            animateY(800)
            invalidate()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
