package com.airguard.ai.ui.settings

import android.app.AlertDialog
import android.os.Bundle
import android.view.*
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.airguard.ai.databinding.FragmentSettingsBinding
import com.airguard.ai.data.BootReceiver

class SettingsFragment : Fragment() {

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!
    private val viewModel: SettingsViewModel by viewModels()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        observeViewModel()
        setupClickListeners()
    }

    private fun observeViewModel() {
        viewModel.lastSync.observe(viewLifecycleOwner) { ts ->
            binding.tvLastSyncSetting.text = if (ts == 0L) "Never synced" else "Last: $ts"
        }
        viewModel.cacheSize.observe(viewLifecycleOwner) { size ->
            binding.tvCacheSize.text = "$size readings stored locally"
        }
        viewModel.syncMessage.observe(viewLifecycleOwner) { msg ->
            if (msg != null) Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupClickListeners() {
        binding.itemSyncNow.setOnClickListener {
            viewModel.syncNow()
        }

        binding.switchAutoSync.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                BootReceiver.scheduleSyncWork(requireContext())
                Toast.makeText(requireContext(), "Auto-sync enabled", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(requireContext(), "Auto-sync disabled", Toast.LENGTH_SHORT).show()
            }
        }

        binding.itemClearCache.setOnClickListener {
            AlertDialog.Builder(requireContext())
                .setTitle("Clear Local Cache")
                .setMessage("This will delete all locally stored sensor readings. You can re-sync afterwards.\n\nContinue?")
                .setPositiveButton("Clear") { _, _ ->
                    viewModel.clearCache()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        binding.itemImportCsv.setOnClickListener {
            Toast.makeText(
                requireContext(),
                "CSV import: place a file named 'sensor_data.csv' in the app's files directory and re-open the app.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
