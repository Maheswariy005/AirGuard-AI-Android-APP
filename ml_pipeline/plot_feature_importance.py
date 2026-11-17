# -*- coding: utf-8 -*-
"""
Publication-quality horizontal bar chart of Random Forest feature importances
for the AirGuard AI IEEE research paper.

Uses the exact verified values from the deployed ONNX model
(audit_model_c_opset17.onnx, exported from audit_model_c.pkl).

Source: ml_pipeline/experiments/final_audit/feature_importances.csv
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Exact feature-importance values from the deployed model ──
# (RF class_weight='balanced', 200 trees, random_state=42, cutoff 2026-06-01)
# Sorted descending by importance.
features = [
    "Gas Resistance",
    "Humidity",
    "Altitude",
    "Pressure",
    "Temperature",
    "Dew Point",
    "Heat Index",
]
importances = [
    54.19,
    12.45,
     8.53,
     7.18,
     6.51,
     5.67,
     5.45,
]

# ── Publication-quality figure ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

# Color: single professional blue palette, darkest for highest importance
colors = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5',
          '#64B5F6', '#90CAF9', '#BBDEFB']

y_pos = range(len(features))
bars = ax.barh(y_pos, importances, color=colors, edgecolor='#0D47A1',
               linewidth=0.5, height=0.65)

# Reverse Y axis so the most important feature is at the top
ax.invert_yaxis()
ax.set_yticks(list(y_pos))
ax.set_yticklabels(features, fontsize=10)

# X axis
ax.set_xlabel('Feature Importance (%)', fontsize=11, labelpad=8)
ax.set_xlim(0, 62)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.tick_params(axis='x', labelsize=9)

# Value labels at end of each bar
for bar, val in zip(bars, importances):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', ha='left', fontsize=9, fontweight='bold',
            color='#333333')

# Clean spines — remove top and right
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Light grid on X axis
ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────
out_dir = r"c:\Users\ADMIN\AppData\AirGuardAI"
out_path = os.path.join(out_dir, "feature_importance.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Figure saved: {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024:.1f} KB")
print(f"DPI: 300")
print(f"Format: PNG")
