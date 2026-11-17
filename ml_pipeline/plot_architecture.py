# -*- coding: utf-8 -*-
"""
Publication-quality system architecture diagram for AirGuard AI IEEE paper.
Two-column-width, 300 DPI, white background.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Palette ───────────────────────────────────────────────────
C_SENSOR   = '#B2EBF2'   # light cyan   – sensor hardware
C_CLOUD    = '#E3F2FD'   # light blue   – remote/cloud
C_PROC     = '#E8EAF6'   # blue-grey    – data processing
C_STORE    = '#E8F5E9'   # light green  – local storage / offline
C_ML       = '#EDE7F6'   # light purple – ML / ONNX
C_AQI      = '#FFF3E0'   # light amber  – AQI-derived output
C_MLOUT    = '#F3E5F5'   # light violet – ML output
C_UI       = '#F5F5F5'   # light grey   – UI
C_ONLINE   = '#1565C0'   # dark blue    – online arrows
C_OFFLINE  = '#2E7D32'   # dark green   – offline arrows
C_BOTH     = '#37474F'   # charcoal     – shared flows
C_BOX_EDG  = '#455A64'   # box edge

TITLE_COL  = '#0D47A1'
HEAD_COL   = '#1A237E'
TEXT_COL   = '#1A1A2E'

fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('white')

# ── Helpers ───────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec=C_BOX_EDG, lw=1.2, rad=0.18):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.0,rounding_size={rad}',
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    return p

def txt(x, y, s, fs=8, ha='center', va='center', bold=False, col=TEXT_COL, **kw):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=weight, color=col,
            linespacing=1.4, zorder=4, **kw)

def arrow(x1, y1, x2, y2, col=C_BOTH, lw=1.5, style='->', dashed=False, label='', lfs=6.8):
    ls = '--' if dashed else '-'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                linestyle=ls, connectionstyle='arc3,rad=0.0'),
                zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center', fontsize=lfs,
                color=col, fontstyle='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))

def section_header(x, y, w, h, title, col_face, col_text=HEAD_COL):
    rbox(x, y, w, h, col_face, ec=C_BOX_EDG, lw=1.4)
    txt(x + w/2, y + h/2, title, fs=7.5, bold=True, col=col_text)

def bullet_block(x, y, w, h, header, bullets, fc, hfs=8.5, bfs=7.5):
    rbox(x, y, w, h, fc)
    txt(x + w/2, y + h - 0.21, header, fs=hfs, bold=True, col=HEAD_COL)
    line_h = (h - 0.32) / max(len(bullets), 1)
    for i, b in enumerate(bullets):
        ty = y + h - 0.32 - (i + 0.55) * line_h
        txt(x + 0.18, ty, b, fs=bfs, ha='left', va='center', col=TEXT_COL)

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
txt(6.0, 7.68,
    'AirGuard AI: On-Device Indoor Air-Quality Risk Classification Under Connectivity Constraints',
    fs=9.5, bold=True, col=TITLE_COL)

# ══════════════════════════════════════════════════════════════
# ROW 1  (y = 5.5–7.0)
# 1a. BME680 Sensor Node    1b. Google Sheets     1c. Android Data Processing
# ══════════════════════════════════════════════════════════════

# 1a – BME680
rbox(0.25, 5.45, 2.3, 1.5, C_SENSOR)
txt(1.40, 6.70, 'BME680 Sensor Node', fs=8.5, bold=True, col=HEAD_COL)
for i, s in enumerate([
    u'\u25b8 Temperature',
    u'\u25b8 Relative Humidity',
    u'\u25b8 Atmospheric Pressure',
    u'\u25b8 Gas Resistance',
    u'\u25b8 Recorded AQI',
]):
    txt(0.42, 6.46 - i*0.21, s, fs=7.5, ha='left', va='center')

# 1b – Google Sheets
rbox(2.95, 5.45, 2.3, 1.5, C_CLOUD)
txt(4.10, 6.70, 'Google Sheets', fs=8.5, bold=True, col=HEAD_COL)
txt(4.10, 6.47, 'Remote Sensor Data Source', fs=7.5, col=TEXT_COL)
txt(4.10, 6.26, u'\u25b8 Historical sensor observations', fs=7.3, ha='center')
txt(4.10, 6.07, u'\u25b8 Fetched when internet available', fs=7.3, ha='center')
txt(4.10, 5.88, u'\u25b8 HTTP/HTTPS access via OkHttp', fs=7.3, ha='center')

# 1c – Android Processing
rbox(5.65, 5.45, 3.4, 1.5, C_PROC)
txt(7.35, 6.70, 'Android Data Processing', fs=8.5, bold=True, col=HEAD_COL)
for i, s in enumerate([
    u'\u25b8 Parse & validate sensor readings',
    u'\u25b8 Extract 7 env. features: Temp, Humidity,',
    u'   Pressure, Gas, Altitude, Dew Pt, Heat Index',
    u'\u25b8 Apply StandardScaler parameters',
    u'\u25b8 No AQI passed to ML model',
]):
    txt(5.82, 6.46 - i*0.19, s, fs=7.3, ha='left', va='center')

# 1d – Online / Offline status badge (top right)
rbox(9.45, 6.55, 2.3, 0.90, 'white', ec=C_ONLINE, lw=1.8)
txt(10.60, 7.26, 'Online Path', fs=7.5, bold=True, col=C_ONLINE)
txt(10.60, 7.05, u'\u27a4  Fetch \u2192 Cache \u2192 Classify', fs=7.0, col=C_ONLINE)
rbox(9.45, 5.45, 2.3, 0.90, 'white', ec=C_OFFLINE, lw=1.8)
txt(10.60, 6.16, 'Offline Path', fs=7.5, bold=True, col=C_OFFLINE)
txt(10.60, 5.95, u'\u27a4  Cache \u2192 Classify (cached)', fs=7.0, col=C_OFFLINE)

# ══════════════════════════════════════════════════════════════
# ROW 2  (y = 3.5–5.1)
# 2a. Local JSON Cache    2b. ONNX ML block    2c. Classification outputs
# ══════════════════════════════════════════════════════════════

# 2a – Local JSON Cache
rbox(0.25, 3.45, 2.8, 1.65, C_STORE)
txt(1.65, 4.86, 'Local JSON Cache', fs=8.5, bold=True, col='#1B5E20')
txt(1.65, 4.65, 'sensor_readings.json', fs=7.8, bold=False, col='#2E7D32',
    fontstyle='italic')
for i, s in enumerate([
    u'\u25b8 Stores latest 2,000 readings',
    u'\u25b8 Available online and offline',
    u'\u25b8 Read by ONNX Runtime for inference',
    u'\u25b8 Updated only when online',
]):
    txt(0.42, 4.44 - i*0.20, s, fs=7.3, ha='left', va='center')

# 2b – ONNX Runtime / ML
rbox(3.45, 3.45, 3.3, 1.65, C_ML)
txt(5.10, 4.86, 'On-Device Machine Learning', fs=8.5, bold=True, col='#4A148C')
txt(5.10, 4.65, 'ONNX Runtime for Android', fs=7.8, col='#6A1B9A', bold=False)
for i, s in enumerate([
    u'\u25b8 Random Forest: 200 decision trees',
    u'\u25b8 7 env. features (AQI excluded)',
    u'\u25b8 class_weight=balanced, rand_state=42',
    u'\u25b8 No remote inference server',
]):
    txt(3.62, 4.44 - i*0.20, s, fs=7.3, ha='left', va='center')

# 2c – Classification Outputs (two boxes side by side)
# AQI-derived
rbox(7.10, 3.45, 2.3, 1.65, C_AQI)
txt(8.25, 4.86, 'AQI-Derived', fs=8.0, bold=True, col='#E65100')
txt(8.25, 4.67, 'Risk Category', fs=8.0, bold=True, col='#E65100')
for i, s in enumerate([
    u'\u25b8 Input: recorded AQI value',
    u'\u25b8 Deterministic threshold map',
    u'\u25b8 RiskLevel.fromAqi()',
    u'\u25b8 Independent of ML model',
]):
    txt(7.27, 4.46 - i*0.20, s, fs=7.3, ha='left', va='center')

# ML-predicted
rbox(9.60, 3.45, 2.1, 1.65, C_MLOUT)
txt(10.65, 4.86, 'ML-Predicted', fs=8.0, bold=True, col='#6A1B9A')
txt(10.65, 4.67, 'Risk Category', fs=8.0, bold=True, col='#6A1B9A')
for i, s in enumerate([
    u'\u25b8 Input: 7 env. features',
    u'\u25b8 ONNX RF inference',
    u'\u25b8 5 classes (no VERY_UNH)',
    u'\u25b8 Confidence score output',
]):
    txt(9.77, 4.46 - i*0.20, s, fs=7.3, ha='left', va='center')

# ══════════════════════════════════════════════════════════════
# ROW 3  (y = 1.1–3.0)
# Android UI
# ══════════════════════════════════════════════════════════════

rbox(0.25, 1.05, 11.5, 2.0, C_UI)
txt(6.0, 2.84, 'Android User Interface (Dashboard & Analysis Screens)', fs=9, bold=True, col=HEAD_COL)

# Dashboard sub-box
rbox(0.5, 1.2, 3.5, 1.55, '#ECEFF1', ec='#78909C', lw=1.0)
txt(2.25, 2.56, 'Dashboard Screen', fs=8.0, bold=True, col='#37474F')
txt(2.25, 2.37, u'\u25b8 Live AQI & sensor readings', fs=7.3, ha='center')
txt(2.25, 2.19, u'\u25b8 AQI-derived risk badge', fs=7.3, ha='center')
txt(2.25, 2.01, u'\u25b8 ML-predicted risk (ONNX)', fs=7.3, ha='center')
txt(2.25, 1.83, u'\u25b8 24-hour AQI trend chart', fs=7.3, ha='center')
txt(2.25, 1.65, u'\u25b8 Online/offline sync status', fs=7.3, ha='center')
txt(2.25, 1.47, u'\u25b8 Agreement / divergence note', fs=7.3, ha='center')

# Comparison sub-box
rbox(4.3, 1.2, 3.1, 1.55, '#ECEFF1', ec='#78909C', lw=1.0)
txt(5.85, 2.56, 'Classification Comparison', fs=8.0, bold=True, col='#37474F')
txt(5.85, 2.37, u'\u25b8 AQI-Derived | ML-Predicted', fs=7.5, ha='center',
    col='#E65100')
txt(5.85, 2.14, u'\u2714  Agree \u2192 green confirmation', fs=7.3, ha='center')
txt(5.85, 1.96, u'\u2718  Differ \u2192 orange explanation', fs=7.3, ha='center')
txt(5.85, 1.73, u'VERY_UNHEALTHY: not predicted', fs=7.0, ha='center',
    col='#9E9E9E', fontstyle='italic')
txt(5.85, 1.52, u'(0 training samples in dataset)', fs=7.0, ha='center',
    col='#9E9E9E', fontstyle='italic')

# Analysis sub-box
rbox(7.65, 1.2, 3.85, 1.55, '#ECEFF1', ec='#78909C', lw=1.0)
txt(9.575, 2.56, 'Analysis Screen', fs=8.0, bold=True, col='#37474F')
txt(9.575, 2.37, u'\u25b8 Latest ML prediction & confidence', fs=7.3, ha='center')
txt(9.575, 2.19, u'\u25b8 Top feature importances (ONNX model)', fs=7.3, ha='center')
txt(9.575, 2.01, u'\u25b8 7 raw sensor feature values', fs=7.3, ha='center')
txt(9.575, 1.83, u'\u25b8 Risk-level distribution (AQI)', fs=7.3, ha='center')
txt(9.575, 1.65, u'\u25b8 Temperature & humidity trends', fs=7.3, ha='center')
txt(9.575, 1.47, u'\u25b8 Risk level history chart', fs=7.3, ha='center')

# ══════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════

# Sensor → Google Sheets
arrow(2.55, 6.20, 2.95, 6.20, col=C_BOTH)
# Google Sheets → Android Processing (online)
arrow(5.25, 6.20, 5.65, 6.20, col=C_ONLINE, label='Online fetch')

# Android Processing → Local JSON Cache  (store)
arrow(7.35, 5.45, 4.0, 5.11, col=C_ONLINE, label='Store')

# Android Processing → AQI Threshold (direct, no ML)
arrow(7.35, 5.45, 8.25, 5.10, col=C_BOTH, label='AQI value')

# Local JSON → ONNX (online read)
arrow(2.95, 4.26, 3.45, 4.26, col=C_ONLINE, label='Read (online)')
# Local JSON → ONNX (offline)
arrow(2.95, 3.95, 3.45, 3.95, col=C_OFFLINE, label='Read (offline)', dashed=True)

# ONNX → ML-predicted output
arrow(6.75, 4.26, 9.60, 4.26, col='#6A1B9A', label='Predict')

# AQI-derived output → UI
arrow(8.25, 3.45, 5.5, 3.05, col='#E65100')
# ML-predicted output → UI
arrow(10.65, 3.45, 7.5, 3.05, col='#6A1B9A')

# Online path annotation
ax.annotate('', xy=(0.65, 5.45), xytext=(0.65, 5.11),
            arrowprops=dict(arrowstyle='->', color=C_OFFLINE, lw=1.4, linestyle='--'),
            zorder=3)
txt(1.30, 5.28, 'Cached data\n(offline use)', fs=6.8, col=C_OFFLINE)

# ══════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════
legend_x, legend_y = 9.55, 2.95
rbox(legend_x - 0.1, legend_y - 0.08, 2.3, 0.65, 'white', ec='#90A4AE', lw=0.8)
txt(legend_x + 1.05, legend_y + 0.46, 'Legend', fs=7.5, bold=True, col='#37474F')
items = [
    (C_ONLINE,  'solid',  'Online data flow'),
    (C_OFFLINE, 'dashed', 'Offline data flow'),
    (C_BOTH,    'solid',  'Shared data flow'),
]
for i, (col, ls, lab) in enumerate(items):
    yl = legend_y + 0.30 - i * 0.17
    lx = legend_x
    ax.plot([lx, lx + 0.4], [yl, yl], color=col, lw=1.6,
            linestyle='--' if ls == 'dashed' else '-', zorder=4)
    ax.annotate('', xy=(lx + 0.4, yl), xytext=(lx + 0.25, yl),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.3), zorder=5)
    txt(lx + 0.5, yl, lab, fs=7.0, ha='left', va='center', col=TEXT_COL)

# ══════════════════════════════════════════════════════════════
# BOUNDARY LINES between major zones
# ══════════════════════════════════════════════════════════════
for yline in [5.42, 3.42, 3.03]:
    ax.plot([0.1, 11.85], [yline, yline], color='#B0BEC5', lw=0.7, linestyle='-', zorder=1)

# Zone labels (left margin)
for ypos, label in [(6.20, 'Data\nAcquisition'),
                    (4.27, 'Processing\n& Inference'),
                    (2.05, 'User\nInterface')]:
    txt(0.08, ypos, label, fs=6.8, ha='center', bold=True, col='#607D8B', rotation=90)

plt.tight_layout(pad=0)
out_path = r"c:\Users\ADMIN\AppData\AirGuardAI\airguard_architecture.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Saved: {out_path}")
print(f"Size : {os.path.getsize(out_path)/1024:.1f} KB")
