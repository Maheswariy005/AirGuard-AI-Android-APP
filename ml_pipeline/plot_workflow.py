# -*- coding: utf-8 -*-
"""
Publication-quality workflow diagram for AirGuard AI IEEE paper.
Shows online and offline operation paths.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colors ───────────────────────────────────────────────────
ONLINE_BG  = '#E3F2FD'   # light blue
OFFLINE_BG = '#FFF3E0'   # light orange
ONLINE_ACCENT  = '#1565C0'
OFFLINE_ACCENT = '#E65100'
BOX_FACE  = 'white'
BOX_EDGE  = '#37474F'
TEXT_DARK = '#1A1A2E'
ARROW_ONLINE  = '#1565C0'
ARROW_OFFLINE = '#E65100'
SHARED_COLOR = '#37474F'

fig, (ax_online, ax_offline) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=300)
fig.patch.set_facecolor('white')

def draw_box(ax, x, y, w, h, text, fontsize=9, facecolor='white', edgecolor=BOX_EDGE, bold=False, text_color=TEXT_DARK):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=weight, zorder=3,
            linespacing=1.3)

def draw_arrow(ax, x1, y1, x2, y2, color, label='', label_offset=(0, 0)):
    """Draw a directional arrow with optional label."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=14,
                            color=color, linewidth=1.5, zorder=1)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=7.5,
                color=color, fontstyle='italic', zorder=3,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9))

# ══════════════════════════════════════════════════════════════
# LEFT PANEL: ONLINE OPERATION
# ══════════════════════════════════════════════════════════════
ax = ax_online
ax.set_xlim(0, 4.2)
ax.set_ylim(0, 6.2)
ax.set_aspect('equal')
ax.axis('off')

# Background panel
bg = FancyBboxPatch((0.05, 0.05), 4.1, 6.1, boxstyle='round,pad=0.15',
                     facecolor=ONLINE_BG, edgecolor=ONLINE_ACCENT, linewidth=1.5, alpha=0.4, zorder=0)
ax.add_patch(bg)
ax.text(2.1, 5.85, 'Online Operation', ha='center', va='center',
        fontsize=12, fontweight='bold', color=ONLINE_ACCENT, zorder=3)

# Boxes (x, y, w, h)
bw, bh = 2.6, 0.7
cx = (4.2 - bw) / 2  # center x

# Google Sheets (top)
draw_box(ax, cx, 4.7, bw, bh, 'Google Sheets\n(Remote Sensor Data)', fontsize=9, bold=True)

# Android App
draw_box(ax, cx, 3.5, bw, bh, 'Android App\n(Fetch Readings)', fontsize=9, bold=True)

# Local JSON Cache
draw_box(ax, cx, 2.3, bw, bh, 'Local JSON Cache\nsensor_readings.json\n(latest 2,000 readings)', fontsize=8)

# ONNX Runtime
draw_box(ax, cx, 1.1, bw, bh, 'ONNX Runtime\nRandom Forest\nClassification', fontsize=9, bold=True)

# Arrows
draw_arrow(ax, 2.1, 4.7, 2.1, 4.2, ARROW_ONLINE, 'HTTP fetch')
draw_arrow(ax, 2.1, 3.5, 2.1, 3.0, ARROW_ONLINE, 'Store')
draw_arrow(ax, 2.1, 2.3, 2.1, 1.8, ARROW_ONLINE, 'Read')

# Network indicator
ax.text(0.3, 5.85, '\u25CF', fontsize=10, color='#4CAF50', zorder=3)
ax.text(0.5, 5.85, 'Network: Available', fontsize=8, color='#4CAF50', va='center', zorder=3)


# ══════════════════════════════════════════════════════════════
# RIGHT PANEL: OFFLINE OPERATION
# ══════════════════════════════════════════════════════════════
ax = ax_offline
ax.set_xlim(0, 4.2)
ax.set_ylim(0, 6.2)
ax.set_aspect('equal')
ax.axis('off')

# Background panel
bg = FancyBboxPatch((0.05, 0.05), 4.1, 6.1, boxstyle='round,pad=0.15',
                     facecolor=OFFLINE_BG, edgecolor=OFFLINE_ACCENT, linewidth=1.5, alpha=0.4, zorder=0)
ax.add_patch(bg)
ax.text(2.1, 5.85, 'Offline Operation', ha='center', va='center',
        fontsize=12, fontweight='bold', color=OFFLINE_ACCENT, zorder=3)

# Google Sheets (greyed out — unavailable)
draw_box(ax, cx, 4.7, bw, bh, 'Google Sheets\n(Unavailable)', fontsize=9, bold=False,
         facecolor='#F5F5F5', edgecolor='#BDBDBD', text_color='#9E9E9E')

# Dashed line from Google Sheets showing disconnection
ax.plot([2.1, 2.1], [4.7, 4.2], color='#BDBDBD', linewidth=1.5, linestyle='--', zorder=1)
ax.text(2.1, 4.45, '\u2715', ha='center', va='center', fontsize=14, color='#F44336', zorder=3,
        bbox=dict(boxstyle='circle,pad=0.15', facecolor='white', edgecolor='#F44336', linewidth=1))

# Local JSON Cache
draw_box(ax, cx, 3.0, bw, bh, 'Local JSON Cache\nsensor_readings.json\n(cached readings)', fontsize=8)

# ONNX Runtime
draw_box(ax, cx, 1.1, bw, bh, 'ONNX Runtime\nRandom Forest\nClassification', fontsize=9, bold=True)

# Arrows
draw_arrow(ax, 2.1, 3.0, 2.1, 1.8, ARROW_OFFLINE, 'Read & classify')

# Network indicator
ax.text(0.3, 5.85, '\u25CF', fontsize=10, color='#F44336', zorder=3)
ax.text(0.5, 5.85, 'Network: Unavailable', fontsize=8, color='#F44336', va='center', zorder=3)

# Note at bottom
ax.text(2.1, 0.35, 'No new readings retrieved\nfrom Google Sheets', ha='center', va='center',
        fontsize=7.5, color='#9E9E9E', fontstyle='italic', zorder=3)

plt.tight_layout(pad=0.5)

out_path = os.path.join(r"c:\Users\ADMIN\AppData\AirGuardAI", "online_offline_workflow.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Figure saved: {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024:.1f} KB")
print(f"DPI: 300")
