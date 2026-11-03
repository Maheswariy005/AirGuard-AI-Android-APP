package com.airguard.ai.ui.readings

import android.os.Bundle
import android.view.*
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.airguard.ai.databinding.FragmentReadingsBinding

class ReadingsFragment : Fragment() {

    private var _binding: FragmentReadingsBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ReadingsViewModel by viewModels()
    private lateinit var adapter: ReadingsAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentReadingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        observeViewModel()
        setupChips()
    }

    private fun setupRecyclerView() {
        adapter = ReadingsAdapter()
        binding.rvReadings.layoutManager = LinearLayoutManager(requireContext())
        binding.rvReadings.adapter = adapter
    }

    private fun observeViewModel() {
        viewModel.readings.observe(viewLifecycleOwner) { readings ->
            adapter.submitList(readings)
            if (readings.isEmpty()) {
                binding.rvReadings.visibility = View.GONE
                binding.emptyState.visibility = View.VISIBLE
            } else {
                binding.rvReadings.visibility = View.VISIBLE
                binding.emptyState.visibility = View.GONE
            }
        }

        viewModel.readingCount.observe(viewLifecycleOwner) { count ->
            binding.tvReadingsCount.text = "$count readings"
        }
    }

    private fun setupChips() {
        binding.chipAll.setOnClickListener { viewModel.setFilter(ReadingsFilter.ALL) }
        binding.chipToday.setOnClickListener { viewModel.setFilter(ReadingsFilter.TODAY) }
        binding.chipWeek.setOnClickListener { viewModel.setFilter(ReadingsFilter.WEEK) }
        binding.chipMonth.setOnClickListener { viewModel.setFilter(ReadingsFilter.MONTH) }
        binding.btnSyncEmpty.setOnClickListener { viewModel.sync() }
        binding.btnFilter.setOnClickListener { viewModel.sync() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
