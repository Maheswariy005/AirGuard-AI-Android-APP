package com.airguard.ai.ui.readings

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.airguard.ai.data.SensorReading
import com.airguard.ai.databinding.ItemReadingBinding

class ReadingsAdapter : ListAdapter<SensorReading, ReadingsAdapter.ViewHolder>(DIFF) {

    inner class ViewHolder(private val binding: ItemReadingBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(reading: SensorReading) {
            binding.tvDatetime.text = reading.dateTime
            binding.tvAqiBadge.text = "AQI ${reading.aqi}"
            binding.tvRiskChip.text = reading.riskLevel.label
            binding.tvTemp.text = "%.1f°C".format(reading.temperature)
            binding.tvHumidity.text = "%.1f%%".format(reading.humidity)
            binding.tvPressure.text = "%.1f hPa".format(reading.pressure)
            binding.tvGas.text = "%.1f".format(reading.gasKOhms)
            try {
                val color = Color.parseColor(reading.riskLevel.colorHex)
                binding.tvAqiBadge.setBackgroundColor(color)
                binding.tvRiskChip.setBackgroundColor(color)
            } catch (_: Exception) {}
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemReadingBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    companion object {
        val DIFF = object : DiffUtil.ItemCallback<SensorReading>() {
            override fun areItemsTheSame(a: SensorReading, b: SensorReading) =
                a.date == b.date && a.time == b.time
            override fun areContentsTheSame(a: SensorReading, b: SensorReading) = a == b
        }
    }
}
