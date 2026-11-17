# -*- coding: utf-8 -*-
"""
AirGuard AI — Revised system architecture diagram (v2).
Corrections applied:
  1. BME680 renamed; AQI removed from sensor measurements;
     AQI shown as a separate reference value from Google Sheets dataset.
  2. Offline dashed arrow goes FROM Local JSON Cache TOWARD ONNX/processing.
  3. AQI-Derived and ML-Predicted outputs each have their own independent arrow to the UI.
  4. Online / offline paths clearly separated.
  5. UI section lists only implemented features.
  6. Technical labels accurate throughout.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out_path = r"c:\Users\ADMIN\AppData\AirGuardAI\airguard_architecture.png"

# ── Palette (unchanged from v1) ───────────────────────────────
C_SENSOR  = '#B2EBF2'   # light cyan
C_CLOUD   = '#E3F2FD'   # light blue
C_PROC    = '#E8EAF6'   # blue-grey
C_STORE   = '#E8F5E9'   # light green
C_ML      = '#EDE7F6'   # light purple
C_AQI     = '#FFF3E0'   # light amber
C_MLOUT   = '#F3E5F5'   # light violet
C_UI      = '#F5F5F5'   # light grey
C_ONLINE  = '#1565C0'   # dark blue
C_OFFLINE = '#2E7D32'   # dark green
C_BOTH    = '#37474F'   # charcoal
C_BOX_EDG = '#455A64'
HEAD_COL  = '#1A237E'
TITLE_COL = '#0D47A1'
TEXT_COL  = '#1A1A2E'

fig = plt.figure(figsize=(12, 8.4), dpi=300, facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 8.4)
ax.axis('off')
ax.set_facecolor('white')

# ── Primitives ────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec=C_BOX_EDG, lw=1.2, rad=0.16):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={rad}',
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)

def txt(x, y, s, fs=8, ha='center', va='center', bold=False,
        col=TEXT_COL, italic=False, **kw):
    weight = 'bold' if bold else 'normal'
    style  = 'italic' if italic else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=weight,
            fontstyle=style, color=col, linespacing=1.35, zorder=4, **kw)

def arrow(x1, y1, x2, y2, col=C_BOTH, lw=1.5, dashed=False, label='', lfs=6.8,
          loffx=0, loffy=0, rad=0.0):
    ls = '--' if dashed else '-'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw, linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=3)
    if label:
        mx = (x1+x2)/2 + loffx
        my = (y1+y2)/2 + loffy
        ax.text(mx, my, label, ha='center', va='center', fontsize=lfs,
                color=col, fontstyle='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.92))

def hline(y, x0=0.1, x1=11.9, col='#B0BEC5'):
    ax.plot([x0, x1], [y, y], color=col, lw=0.7, zorder=1)

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
txt(6.0, 8.14,
    'AirGuard AI: On-Device Indoor Air-Quality Risk Classification Under Connectivity Constraints',
    fs=9.5, bold=True, col=TITLE_COL)

# ══════════════════════════════════════════════════════════════
# ROW 1  y = 5.75 – 7.65   DATA ACQUISITION
# ══════════════════════════════════════════════════════════════
hline(7.68)
txt(0.08, 6.70, 'Data\nAcquisition', fs=6.8, bold=True, col='#607D8B', rotation=90)

# ── 1A. BME680 Sensor Measurements ───────────────────────────
rbox(0.28, 5.78, 2.4, 1.72, C_SENSOR)
txt(1.48, 7.28, 'BME680 Sensor', fs=8.8, bold=True, col=HEAD_COL)
txt(1.48, 7.09, 'Measurements', fs=8.8, bold=True, col=HEAD_COL)
for i, s in enumerate([
    u'\u25b8 Temperature',
    u'\u25b8 Relative Humidity',
    u'\u25b8 Atmospheric Pressure',
    u'\u25b8 Gas Resistance',
]):
    txt(0.46, 6.88 - i*0.24, s, fs=7.5, ha='left', va='center')

# ── 1B. Google Sheets ─────────────────────────────────────────
rbox(3.00, 5.78, 2.65, 1.72, C_CLOUD)
txt(4.325, 7.30, 'Google Sheets', fs=8.8, bold=True, col=HEAD_COL)
txt(4.325, 7.10, 'Remote Sensor Dataset', fs=7.6, col=TEXT_COL)
for i, s in enumerate([
    u'\u25b8 Historical sensor observations',
    u'\u25b8 Fetched when internet available',
    u'\u25b8 Provides recorded AQI values',
    u'\u25b8 HTTP access via OkHttp',
]):
    txt(3.18, 6.88 - i*0.24, s, fs=7.3, ha='left', va='center')

# ── Recorded AQI reference badge (inside Google Sheets box) ──
rbox(3.08, 5.82, 1.3, 0.38, '#FFF8E1', ec='#FFA000', lw=1.0, rad=0.10)
# move to avoid overlap — place it right of description text at bottom
rbox(4.42, 5.82, 1.15, 0.38, '#FFF8E1', ec='#FFA000', lw=1.0, rad=0.10)
txt(5.00, 6.01, 'Recorded AQI\n(reference value)', fs=6.5, bold=False, col='#E65100')

# ── 1C. Android Data Processing ──────────────────────────────
rbox(6.00, 5.78, 3.55, 1.72, C_PROC)
txt(7.775, 7.30, 'Android Data Processing', fs=8.8, bold=True, col=HEAD_COL)
for i, s in enumerate([
    u'\u25b8 Parse & validate sensor readings',
    u'\u25b8 7 env. features: Temp, Humidity, Pressure,',
    u'   Gas, Altitude, Dew Point, Heat Index',
    u'\u25b8 Apply StandardScaler parameters',
    u'\u25b8 AQI not passed to ML model',
]):
    txt(6.17, 7.09 - i*0.22, s, fs=7.3, ha='left', va='center')

# ── 1D. Online / Offline status badges ───────────────────────
rbox(9.90, 7.00, 1.85, 0.68, 'white', ec=C_ONLINE, lw=1.6)
txt(10.825, 7.34, 'Online Path', fs=7.5, bold=True, col=C_ONLINE)
txt(10.825, 7.13, u'\u27a4 Sheets \u2192 Cache \u2192 ONNX', fs=6.8, col=C_ONLINE)

rbox(9.90, 6.14, 1.85, 0.68, 'white', ec=C_OFFLINE, lw=1.6)
txt(10.825, 6.48, 'Offline Path', fs=7.5, bold=True, col=C_OFFLINE)
txt(10.825, 6.27, u'\u27a4 Cache \u2192 ONNX (cached)', fs=6.8, col=C_OFFLINE)

# ══════════════════════════════════════════════════════════════
# ROW 2  y = 3.55 – 5.50   PROCESSING & INFERENCE
# ══════════════════════════════════════════════════════════════
hline(5.55)
txt(0.08, 4.52, 'Processing\n& Inference', fs=6.8, bold=True, col='#607D8B', rotation=90)

# ── 2A. Local JSON Cache ─────────────────────────────────────
rbox(0.28, 3.58, 2.65, 1.72, C_STORE)
txt(1.605, 5.07, 'Local JSON Cache', fs=8.8, bold=True, col='#1B5E20')
txt(1.605, 4.85, 'sensor_readings.json', fs=7.6, col='#2E7D32', italic=True)
for i, s in enumerate([
    u'\u25b8 Stores latest 2,000 readings',
    u'\u25b8 Available online and offline',
    u'\u25b8 Read by ONNX Runtime',
    u'\u25b8 Updated only when online',
]):
    txt(0.46, 4.63 - i*0.24, s, fs=7.3, ha='left', va='center')

# ── 2B. ONNX Runtime / ML ────────────────────────────────────
rbox(3.30, 3.58, 3.3, 1.72, C_ML)
txt(4.95, 5.07, 'On-Device Machine Learning', fs=8.8, bold=True, col='#4A148C')
txt(4.95, 4.85, 'ONNX Runtime for Android', fs=7.8, col='#6A1B9A')
for i, s in enumerate([
    u'\u25b8 Random Forest: 200 decision trees',
    u'\u25b8 7 env. features (AQI excluded)',
    u'\u25b8 class_weight=balanced, seed=42',
    u'\u25b8 No remote inference server',
]):
    txt(3.47, 4.63 - i*0.24, s, fs=7.3, ha='left', va='center')

# ── 2C. AQI-Derived output ───────────────────────────────────
rbox(6.95, 3.58, 2.3, 1.72, C_AQI)
txt(8.10, 5.07, 'AQI-Derived', fs=8.3, bold=True, col='#E65100')
txt(8.10, 4.86, 'Risk Category', fs=8.3, bold=True, col='#E65100')
for i, s in enumerate([
    u'\u25b8 Input: recorded AQI value',
    u'\u25b8 Deterministic threshold map',
    u'\u25b8 RiskLevel.fromAqi()',
    u'\u25b8 Independent of ML model',
]):
    txt(7.12, 4.65 - i*0.24, s, fs=7.3, ha='left', va='center')

# ── 2D. ML-Predicted output ──────────────────────────────────
rbox(9.52, 3.58, 2.25, 1.72, C_MLOUT)
txt(10.645, 5.07, 'ML-Predicted', fs=8.3, bold=True, col='#6A1B9A')
txt(10.645, 4.86, 'Risk Category', fs=8.3, bold=True, col='#6A1B9A')
for i, s in enumerate([
    u'\u25b8 Input: 7 env. features',
    u'\u25b8 ONNX RF inference',
    u'\u25b8 5 output classes',
    u'\u25b8 Confidence score',
]):
    txt(9.69, 4.65 - i*0.24, s, fs=7.3, ha='left', va='center')

# ══════════════════════════════════════════════════════════════
# ROW 3  y = 0.90 – 3.20   USER INTERFACE
# ══════════════════════════════════════════════════════════════
hline(3.30)
txt(0.08, 2.08, 'User\nInterface', fs=6.8, bold=True, col='#607D8B', rotation=90)

rbox(0.28, 0.88, 11.44, 2.22, C_UI)
txt(6.0, 2.90, 'Android User Interface — Dashboard & Analysis Screens', fs=9, bold=True, col=HEAD_COL)

# Dashboard sub-panel
rbox(0.52, 0.98, 3.4, 1.72, '#ECEFF1', ec='#78909C', lw=1.0)
txt(2.22, 2.49, 'Dashboard Screen', fs=8.0, bold=True, col='#37474F')
for i, s in enumerate([
    u'\u25b8 Live AQI & sensor readings',
    u'\u25b8 AQI-derived risk badge',
    u'\u25b8 ML-predicted risk (ONNX)',
    u'\u25b8 Online/offline sync status',
    u'\u25b8 24-hour AQI trend chart',
]):
    txt(0.70, 2.28 - i*0.24, s, fs=7.3, ha='left', va='center')

# Comparison sub-panel
rbox(4.18, 0.98, 3.0, 1.72, '#ECEFF1', ec='#78909C', lw=1.0)
txt(5.68, 2.49, 'Classification Comparison', fs=8.0, bold=True, col='#37474F')
txt(5.68, 2.27, u'AQI-Derived  |  ML-Predicted', fs=7.8, ha='center', col='#E65100')
txt(5.68, 2.07, u'\u2714  Agree \u2192 green confirmation', fs=7.3, ha='center')
txt(5.68, 1.88, u'\u2718  Differ \u2192 orange explanation', fs=7.3, ha='center')
txt(5.68, 1.65, u'VERY_UNHEALTHY: not predicted', fs=6.9, ha='center',
    col='#9E9E9E', italic=True)
txt(5.68, 1.46, u'(0 training samples in dataset)', fs=6.9, ha='center',
    col='#9E9E9E', italic=True)

# Analysis sub-panel
rbox(7.44, 0.98, 4.1, 1.72, '#ECEFF1', ec='#78909C', lw=1.0)
txt(9.49, 2.49, 'Analysis Screen', fs=8.0, bold=True, col='#37474F')
for i, s in enumerate([
    u'\u25b8 Latest ML prediction & confidence',
    u'\u25b8 Top feature importances (ONNX model)',
    u'\u25b8 7 raw sensor feature values',
    u'\u25b8 Risk-level distribution (AQI-based)',
    u'\u25b8 Temperature & humidity trend chart',
]):
    txt(7.62, 2.28 - i*0.24, s, fs=7.3, ha='left', va='center')

# ══════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════

# 1. Sensor → Google Sheets (readings feed the dataset)
arrow(2.68, 6.64, 3.00, 6.64, col=C_BOTH, label='Sensor\ndata')

# 2. Google Sheets → Android Processing  (online fetch)
arrow(5.65, 6.64, 6.00, 6.64, col=C_ONLINE, label='Online fetch')

# 3. Android Processing → Local JSON Cache  (store)
arrow(7.78, 5.78, 7.78, 5.42, col=C_BOTH)   # down then left
arrow(7.78, 5.42, 1.61, 5.30, col=C_ONLINE, label='Store (online)', loffx=0, loffy=0.12)

# 4. Local JSON Cache → ONNX (online read)
arrow(2.93, 4.44, 3.30, 4.44, col=C_ONLINE, label='Read')

# 5. Local JSON Cache → ONNX (offline read) — CORRECTED: flows from cache toward ONNX
arrow(2.93, 4.08, 3.30, 4.08, col=C_OFFLINE, dashed=True, label='Read (cached)')

# 6. ONNX → ML-Predicted output
arrow(6.60, 4.44, 9.52, 4.44, col='#6A1B9A', label='Classify')

# 7. AQI from Google Sheets → AQI-Derived output  (independent, no ML)
arrow(4.90, 5.78, 7.78, 5.30, col='#E65100', rad=0.0)
arrow(7.78, 5.30, 8.10, 5.30, col='#E65100')

# 8. AQI-Derived → UI  (independent arrow)
arrow(8.10, 3.58, 5.68, 3.20, col='#E65100', label='AQI category', loffx=0.5, loffy=0.12)

# 9. ML-Predicted → UI  (independent arrow, different path)
arrow(10.645, 3.58, 8.80, 3.20, col='#6A1B9A', label='ML category', loffx=-0.4, loffy=0.12)

# 10. No-connectivity blocker on offline path (X mark)
ax.text(4.33, 6.64, u'\u2715', ha='center', va='center', fontsize=14,
        color='#F44336', fontweight='bold', zorder=5,
        bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec='#F44336', lw=1.2))
txt(4.33, 6.33, 'No remote access\nwhen offline', fs=6.5, col='#F44336', italic=True)

# ══════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════
rbox(9.52, 1.08, 2.24, 0.78, 'white', ec='#90A4AE', lw=0.8)
txt(10.64, 1.72, 'Legend', fs=7.5, bold=True, col='#37474F')
legend_items = [
    (C_ONLINE,  False, 'Online data flow'),
    (C_OFFLINE, True,  'Offline data flow'),
    (C_BOTH,    False, 'Shared / both paths'),
]
for i, (col, dashed, lab) in enumerate(legend_items):
    yl = 1.56 - i * 0.19
    lx = 9.60
    ax.plot([lx, lx + 0.38], [yl, yl], color=col, lw=1.6,
            linestyle='--' if dashed else '-', zorder=4)
    ax.annotate('', xy=(lx + 0.38, yl), xytext=(lx + 0.22, yl),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.3), zorder=5)
    txt(lx + 0.48, yl, lab, fs=6.9, ha='left', va='center', col=TEXT_COL)

# ══════════════════════════════════════════════════════════════
# ZONE SEPARATORS
# ══════════════════════════════════════════════════════════════
hline(7.73)
hline(5.55)
hline(3.30)

# ── Save ──────────────────────────────────────────────────────
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved : {out_path}")
print(f"Size  : {os.path.getsize(out_path)/1024:.1f} KB")
