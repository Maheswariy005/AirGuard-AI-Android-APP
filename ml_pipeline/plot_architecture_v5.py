# -*- coding: utf-8 -*-
"""
AirGuard AI — Architecture diagram v5.
Targeted corrections over v4:

  1. Arrow H (ONNX -> ML-Predicted) rerouted to travel BELOW the AQI block
     instead of through it, eliminating the visual appearance of a connection
     between AQI-Derived and ML-Predicted outputs.

  2. All four UI charts verified against current source:
     - DashboardFragment.kt: setupChart() / updateChart() -> binding.chartTrend  OK
     - AnalysisFragment.kt: updatePieChart() -> binding.chartAqiDistribution     OK
     - AnalysisFragment.kt: updateRiskHistoryChart() -> binding.chartRiskHistory  OK
     - AnalysisFragment.kt: updateCombinedChart() -> binding.chartCombined        OK
     All four confirmed implemented. UI panel text unchanged.

Everything else identical to v4.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = r"c:\Users\ADMIN\AppData\AirGuardAI\airguard_architecture.png"

CS  = '#B2EBF2'; CC  = '#E3F2FD'; CP  = '#E8EAF6'; CG  = '#E8F5E9'
CM  = '#EDE7F6'; CA  = '#FFF3E0'; CMO = '#F3E5F5'; CUI = '#F5F5F5'
CON = '#1565C0'; COF = '#2E7D32'; CSH = '#37474F'; CBE = '#455A64'
HCol = '#1A237E'; TCol = '#0D47A1'; TC = '#1A1A2E'

W, H = 12.4, 8.6
fig = plt.figure(figsize=(W, H), dpi=300, facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off'); ax.set_facecolor('white')

def rbox(x, y, w, h, fc, ec=CBE, lw=1.2, r=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

def t(x, y, s, fs=8, ha='center', va='center', bold=False,
      col=TC, italic=False, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            color=col, linespacing=1.35, zorder=4, **kw)

def arr(x1, y1, x2, y2, col=CSH, lw=1.5, dash=False,
        label='', lfs=6.8, dx=0, dy=0, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                linestyle='--' if dash else '-',
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=3)
    if label:
        ax.text((x1+x2)/2+dx, (y1+y2)/2+dy, label, ha='center', va='center',
                fontsize=lfs, color=col, fontstyle='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.92))

def route_below(ax, x1, y1, x2, y2, y_pass, col, lw=1.5):
    """
    Draw an elbow arrow from (x1,y1) going DOWN to y_pass,
    then RIGHT to x2, then UP to y2.
    Avoids crossing any block between x1 and x2.
    """
    # segment 1: down
    ax.annotate('', xy=(x1, y_pass), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-', color=col, lw=lw,
                                connectionstyle='arc3,rad=0'), zorder=3)
    # segment 2: right
    ax.annotate('', xy=(x2, y_pass), xytext=(x1, y_pass),
                arrowprops=dict(arrowstyle='-', color=col, lw=lw,
                                connectionstyle='arc3,rad=0'), zorder=3)
    # segment 3: up with arrowhead
    ax.annotate('', xy=(x2, y2), xytext=(x2, y_pass),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle='arc3,rad=0'), zorder=3)
    # label midpoint of horizontal segment
    ax.text((x1+x2)/2, y_pass-0.14, 'Classify', ha='center', va='center',
            fontsize=6.8, color=col, fontstyle='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.92))

def hline(y, x0=0.15, x1=12.25, col='#B0BEC5'):
    ax.plot([x0, x1], [y, y], color=col, lw=0.7, zorder=1)

def zone_label(x, y, s):
    t(x, y, s, fs=6.8, bold=True, col='#607D8B', rotation=90)

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
t(W/2, 8.32,
  'AirGuard AI: On-Device Indoor Air-Quality Risk Classification Under Connectivity Constraints',
  fs=9.5, bold=True, col=TCol)
hline(8.18)

# ═══════════════════════════════════════════════════════════════
# ZONE 1 — DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════
zone_label(0.09, 7.04, 'Data\nAcquisition')
hline(5.85)

# 1A. BME680
rbox(0.28, 5.95, 2.45, 2.00, CS)
t(1.505, 7.72, 'BME680 Sensor', fs=8.8, bold=True, col=HCol)
t(1.505, 7.52, 'Measurements', fs=8.8, bold=True, col=HCol)
for i, s in enumerate([
    u'\u25b8 Temperature',
    u'\u25b8 Relative Humidity',
    u'\u25b8 Atmospheric Pressure',
    u'\u25b8 Gas Resistance',
]):
    t(0.46, 7.31 - i*0.26, s, fs=7.5, ha='left', va='center')

# 1B. Google Sheets
rbox(3.05, 5.95, 2.80, 2.00, CC)
t(4.45, 7.72, 'Google Sheets', fs=8.8, bold=True, col=HCol)
t(4.45, 7.52, 'Remote Sensor Dataset', fs=7.5, col=TC)
for i, s in enumerate([
    u'\u25b8 Historical sensor observations',
    u'\u25b8 Fetched when internet available',
    u'\u25b8 HTTP access via OkHttp',
]):
    t(3.22, 7.28 - i*0.26, s, fs=7.3, ha='left', va='center')
rbox(3.13, 6.00, 2.62, 0.44, '#FFF8E1', ec='#FFA000', lw=1.0, r=0.10)
t(4.44, 6.22, u'Recorded AQI  (reference value from dataset)', fs=7.0, col='#E65100')

# 1C. Android Data Processing
rbox(6.20, 5.95, 3.60, 2.00, CP)
t(8.00, 7.72, 'Android Data Processing', fs=8.8, bold=True, col=HCol)
for i, s in enumerate([
    u'\u25b8 Parse & validate sensor readings',
    u'\u25b8 7 env. features: Temp, Humidity, Pressure,',
    u'   Gas, Altitude, Dew Point, Heat Index',
    u'\u25b8 Apply StandardScaler parameters',
    u'\u25b8 AQI not passed to ML model',
]):
    t(6.37, 7.52 - i*0.26, s, fs=7.3, ha='left', va='center')

# 1D. Online badge
rbox(10.15, 7.30, 2.08, 0.72, 'white', ec=CON, lw=1.6)
t(11.19, 7.66, 'Online Path', fs=7.5, bold=True, col=CON)
t(11.19, 7.46, u'\u27a4 Sheets \u2192 Cache \u2192 ONNX', fs=6.9, col=CON)

# 1E. Offline badge (updated text from v4)
rbox(10.15, 6.22, 2.08, 1.00, 'white', ec=COF, lw=1.6)
t(11.19, 6.98, 'Offline Path', fs=7.5, bold=True, col=COF)
t(11.19, 6.76, u'Previously cached readings;', fs=6.5, col=COF)
t(11.19, 6.57, u'local inference requires', fs=6.5, col=COF)
t(11.19, 6.38, u'no network.', fs=6.5, col=COF)

# ═══════════════════════════════════════════════════════════════
# ZONE 2 — PROCESSING & INFERENCE
# ═══════════════════════════════════════════════════════════════
zone_label(0.09, 4.68, 'Processing\n& Inference')
hline(3.46)

# 2A. Local JSON Cache
rbox(0.28, 3.56, 2.80, 2.04, CG)
t(1.68, 5.38, 'Local JSON Cache', fs=8.8, bold=True, col='#1B5E20')
t(1.68, 5.16, 'sensor_readings.json', fs=7.6, col='#2E7D32', italic=True)
for i, s in enumerate([
    u'\u25b8 Stores latest 2,000 readings',
    u'\u25b8 Available online and offline',
    u'\u25b8 Read by ONNX Runtime',
    u'\u25b8 Updated only when online',
]):
    t(0.46, 4.94 - i*0.26, s, fs=7.3, ha='left', va='center')

# 2B. ONNX Runtime (class_weight and random_state verified from audit_config.json)
rbox(3.42, 3.56, 3.35, 2.04, CM)
t(5.095, 5.38, 'On-Device Machine Learning', fs=8.8, bold=True, col='#4A148C')
t(5.095, 5.16, 'ONNX Runtime for Android', fs=7.8, col='#6A1B9A')
for i, s in enumerate([
    u'\u25b8 Random Forest: 200 trees',
    u'\u25b8 class_weight=balanced, random_state=42',
    u'\u25b8 7 env. features (AQI excluded)',
    u'\u25b8 No remote inference server',
]):
    t(3.59, 4.94 - i*0.26, s, fs=7.3, ha='left', va='center')

# 2C. AQI-Derived output
# Block occupies x: 7.10–9.45, y: 3.56–5.60
rbox(7.10, 3.56, 2.35, 2.04, CA)
t(8.275, 5.38, 'AQI-Derived', fs=8.3, bold=True, col='#E65100')
t(8.275, 5.17, 'Risk Category', fs=8.3, bold=True, col='#E65100')
for i, s in enumerate([
    u'\u25b8 Input: recorded AQI value',
    u'\u25b8 Deterministic threshold map',
    u'\u25b8 RiskLevel.fromAqi()',
    u'\u25b8 Independent of ML model',
]):
    t(7.27, 4.94 - i*0.26, s, fs=7.3, ha='left', va='center')

# 2D. ML-Predicted output
# Block occupies x: 9.72–12.07, y: 3.56–5.60
rbox(9.72, 3.56, 2.35, 2.04, CMO)
t(10.895, 5.38, 'ML-Predicted', fs=8.3, bold=True, col='#6A1B9A')
t(10.895, 5.17, 'Risk Category', fs=8.3, bold=True, col='#6A1B9A')
for i, s in enumerate([
    u'\u25b8 Input: 7 env. features',
    u'\u25b8 ONNX RF inference',
    u'\u25b8 5 output classes',
    u'\u25b8 Confidence score',
]):
    t(9.89, 4.94 - i*0.26, s, fs=7.3, ha='left', va='center')

# ═══════════════════════════════════════════════════════════════
# ZONE 3 — USER INTERFACE
# ═══════════════════════════════════════════════════════════════
zone_label(0.09, 2.08, 'User\nInterface')
rbox(0.28, 0.72, 11.95, 2.54, CUI)
t(W/2, 3.06, 'Android User Interface — Dashboard & Analysis Screens', fs=9.0, bold=True, col=HCol)

# Dashboard (chartTrend confirmed: binding.chartTrend + updateChart() takeLast(24))
rbox(0.52, 0.82, 3.52, 2.00, '#ECEFF1', ec='#78909C', lw=1.0)
t(2.28, 2.60, 'Dashboard Screen', fs=8.0, bold=True, col='#37474F')
for i, s in enumerate([
    u'\u25b8 Live AQI & sensor readings',
    u'\u25b8 24-hour AQI trend (line chart)',
    u'\u25b8 AQI-derived risk badge',
    u'\u25b8 ONNX ML prediction & confidence',
    u'\u25b8 Online / offline sync status',
]):
    t(0.70, 2.38 - i*0.26, s, fs=7.3, ha='left', va='center')

# Classification Comparison
rbox(4.30, 0.82, 3.10, 2.00, '#ECEFF1', ec='#78909C', lw=1.0)
t(5.85, 2.60, 'Classification Comparison', fs=8.0, bold=True, col='#37474F')
t(5.85, 2.37, u'AQI-Derived  |  ML-Predicted', fs=7.8, ha='center', col='#E65100')
t(5.85, 2.14, u'\u2714 Agree \u2192 green confirmation', fs=7.3, ha='center')
t(5.85, 1.92, u'\u2718 Differ \u2192 orange explanation', fs=7.3, ha='center')
t(5.85, 1.67, u'VERY_UNHEALTHY: not predicted', fs=6.9, ha='center',
  col='#9E9E9E', italic=True)
t(5.85, 1.47, u'(0 training samples in dataset)', fs=6.9, ha='center',
  col='#9E9E9E', italic=True)

# Analysis (all three charts confirmed: chartAqiDistribution, chartRiskHistory, chartCombined)
rbox(7.66, 0.82, 4.10, 2.00, '#ECEFF1', ec='#78909C', lw=1.0)
t(9.71, 2.60, 'Analysis Screen', fs=8.0, bold=True, col='#37474F')
for i, s in enumerate([
    u'\u25b8 ML prediction & confidence score',
    u'\u25b8 Top feature importances (ONNX)',
    u'\u25b8 AQI risk-level distribution (pie chart)',
    u'\u25b8 Risk-event history (bar chart)',
    u'\u25b8 Temp. & humidity trend (line chart)',
]):
    t(7.84, 2.38 - i*0.26, s, fs=7.3, ha='left', va='center')

# ═══════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════

# A. BME680 -> Google Sheets
arr(2.73, 6.95, 3.05, 6.95, col=CSH, label='Readings')

# B. Google Sheets -> Android Processing (online fetch)
arr(5.85, 6.95, 6.20, 6.95, col=CON, label='Online fetch')

# D. Android Processing -> Local JSON Cache (store)
arr(8.00, 5.95, 2.70, 5.60, col=CON, label='Store', lfs=6.8, dy=0.14)

# E. Local JSON Cache -> ONNX (online read, solid)
arr(3.08, 4.80, 3.42, 4.80, col=CON, label='Read')

# F. Local JSON Cache -> ONNX (offline read, dashed green)
arr(3.08, 4.38, 3.42, 4.38, col=COF, dash=True, label='Read (cached)')

# G. Recorded AQI badge -> AQI-Derived output (amber, two segments)
arr(5.72, 6.22, 7.10, 5.60, col='#E65100')
arr(7.10, 5.60, 8.28, 5.60, col='#E65100')

# H. ONNX -> ML-Predicted (purple, REROUTED to travel BELOW both output blocks)
# Goes: from ONNX right edge (6.77, 4.70) DOWN to y=3.26 (below the blocks),
# then RIGHT to x=10.895, then UP into ML-Predicted block bottom (y=3.56).
# This path clears the AQI block entirely and no arrow crosses it.
route_below(ax,
    x1=6.77, y1=4.60,   # ONNX right edge, mid-height
    x2=10.895, y2=3.56, # bottom centre of ML-Predicted block
    y_pass=3.26,         # below both output blocks (they start at y=3.56)
    col='#6A1B9A', lw=1.5)

# I. AQI-Derived -> UI (independent amber arrow)
arr(8.28, 3.56, 5.85, 3.32, col='#E65100', label='AQI category', dx=0.6, dy=0.14)

# J. ML-Predicted -> UI (independent purple arrow)
arr(10.895, 3.56, 9.00, 3.32, col='#6A1B9A', label='ML category', dx=-0.5, dy=0.14)

# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
rbox(10.10, 0.86, 2.10, 0.84, 'white', ec='#90A4AE', lw=0.8)
t(11.15, 1.57, 'Legend', fs=7.5, bold=True, col='#37474F')
for i, (col, dsh, lab) in enumerate([
    (CON, False, 'Online data flow'),
    (COF, True,  'Offline data flow'),
    (CSH, False, 'Shared / both paths'),
]):
    yl = 1.40 - i*0.18
    lx = 10.18
    ax.plot([lx, lx+0.38], [yl, yl], color=col, lw=1.6,
            linestyle='--' if dsh else '-', zorder=4)
    ax.annotate('', xy=(lx+0.38, yl), xytext=(lx+0.22, yl),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.3), zorder=5)
    t(lx+0.48, yl, lab, fs=6.9, ha='left', va='center', col=TC)

# Zone separators
hline(8.18); hline(5.85); hline(3.46)

fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved : {OUT}")
print(f"Size  : {os.path.getsize(OUT)/1024:.1f} KB")
