"""
Step 1: HAZARDOUS distribution analysis
========================================
- Loads and cleans the AeroSense Device 2 CSV
- Derives RiskLevel labels using exact AQI thresholds
- Reports HAZARDOUS sample distribution by calendar date
- Reports class distribution per calendar month
- Confirms VERY_UNHEALTHY count is exactly zero
- Reports duplicate counts and final cleaned row count
"""

import sys
import io
import warnings
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────
CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"

FEATURES = [
    "Temperature(°C)",
    "Humidity(%)",
    "Pressure(hPa)",
    "Gas (kOhms)",
    "Altitude(m)",
    "DewPoint(°C)",
    "HeatIndex(°C)",
]
AQI_COL  = "AQI"
DATE_COL = "Date"
TIME_COL = "Time"

ALL_CLASSES = [
    "GOOD",
    "MODERATE",
    "UNHEALTHY_SENSITIVE",
    "UNHEALTHY",
    "VERY_UNHEALTHY",
    "HAZARDOUS",
]

def aqi_to_risk(aqi):
    """Mirror of RiskLevel.fromAqi() — must match exactly."""
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

# ── 1. Load CSV ───────────────────────────────────────────────
print("=" * 70)
print("  Step 1: HAZARDOUS Distribution Analysis")
print("=" * 70)

df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"\n[1] Raw CSV shape     : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"    Columns           : {list(df_raw.columns)}")

# ── 2. Replace 'N/A' strings → NaN ───────────────────────────
df = df_raw.replace("N/A", np.nan).copy()

# ── 3. Count duplicates before dedup ─────────────────────────
n_before_dedup = len(df)
df.drop_duplicates(inplace=True)
n_dup = n_before_dedup - len(df)
print(f"\n[2] Duplicate record count BEFORE dedup : {n_before_dedup}")
print(f"    Duplicate rows removed               : {n_dup}")
print(f"    Rows after dedup                     : {len(df)}")

# ── 4. Convert numeric columns ────────────────────────────────
for col in FEATURES + [AQI_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 5. Drop rows where any feature or AQI is NaN ─────────────
n_before_na = len(df)
df.dropna(subset=FEATURES + [AQI_COL], inplace=True)
n_na = n_before_na - len(df)
print(f"\n[3] Rows dropped (NaN in features or AQI) : {n_na}")
print(f"    Rows remaining after cleaning           : {len(df)}")

# ── 6. Derive labels ──────────────────────────────────────────
df["RiskLevel"] = df[AQI_COL].apply(aqi_to_risk)

# ── 7. Parse datetime ─────────────────────────────────────────
df["_datetime"] = pd.to_datetime(
    df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
    errors="coerce",
)
n_bad_dt = df["_datetime"].isna().sum()
if n_bad_dt > 0:
    print(f"\n    Rows with unparseable datetime dropped : {n_bad_dt}")
    df = df.dropna(subset=["_datetime"])

# Sort chronologically
df.sort_values("_datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

df["_date"] = df["_datetime"].dt.date

print(f"\n    Total rows after all cleaning           : {len(df)}")
print(f"    Timeline : {df['_datetime'].min()}  →  {df['_datetime'].max()}")

# ── 8. Overall class distribution ────────────────────────────
print("\n" + "=" * 70)
print("  OVERALL CLASS DISTRIBUTION")
print("=" * 70)
class_counts = df["RiskLevel"].value_counts()
for cls in ALL_CLASSES:
    cnt = class_counts.get(cls, 0)
    pct = cnt / len(df) * 100
    flag = "  <<< ZERO" if cnt == 0 else ""
    print(f"  {cls:<26} : {cnt:>6}  ({pct:5.1f}%){flag}")

# ── 9. VERY_UNHEALTHY confirmation ───────────────────────────
very_unhealthy_total = class_counts.get("VERY_UNHEALTHY", 0)
print(f"\n[CONFIRM] VERY_UNHEALTHY total count across ALL dates : {very_unhealthy_total}")
if very_unhealthy_total == 0:
    print("          CONFIRMED: zero VERY_UNHEALTHY samples in the dataset.")
    print("          AQI range for VERY_UNHEALTHY would be (80, 90].")
    print(f"          Dataset AQI max = {df[AQI_COL].max():.2f}, min = {df[AQI_COL].min():.2f}")
    aqi_80_90 = df[(df[AQI_COL] > 80) & (df[AQI_COL] <= 90)]
    print(f"          Rows with AQI in (80,90]: {len(aqi_80_90)}")
else:
    print(f"          WARNING: Expected zero but found {very_unhealthy_total}.")

# ── 10. HAZARDOUS samples by date ────────────────────────────
print("\n" + "=" * 70)
print("  HAZARDOUS SAMPLES BY CALENDAR DATE")
print("=" * 70)

haz_df = df[df["RiskLevel"] == "HAZARDOUS"].copy()
total_hazardous = len(haz_df)

if total_hazardous == 0:
    print("\n  No HAZARDOUS samples found in the dataset.")
else:
    # Group by date
    date_groups = df.groupby("_date")

    haz_by_date = []
    for date, group in date_groups:
        haz_count = (group["RiskLevel"] == "HAZARDOUS").sum()
        if haz_count > 0:
            total_that_day = len(group)
            haz_pct = haz_count / total_that_day * 100
            aqi_min = group[AQI_COL].min()
            aqi_max = group[AQI_COL].max()
            haz_by_date.append({
                "Date"        : date,
                "Total"       : total_that_day,
                "HAZARDOUS"   : haz_count,
                "HAZ%"        : haz_pct,
                "AQI_min"     : aqi_min,
                "AQI_max"     : aqi_max,
            })

    haz_table = pd.DataFrame(haz_by_date)

    print(f"\n  {'Date':<14}  {'Total':>7}  {'HAZARDOUS':>9}  {'HAZ%':>7}  {'AQI min':>8}  {'AQI max':>8}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*8}")
    for _, row in haz_table.iterrows():
        print(f"  {str(row['Date']):<14}  {row['Total']:>7}  {row['HAZARDOUS']:>9}  "
              f"{row['HAZ%']:>6.1f}%  {row['AQI_min']:>8.2f}  {row['AQI_max']:>8.2f}")

    first_haz_date   = haz_table["Date"].min()
    last_haz_date    = haz_table["Date"].max()
    n_distinct_dates = len(haz_table)

    print(f"\n  First date HAZARDOUS appears : {first_haz_date}")
    print(f"  Last date HAZARDOUS appears  : {last_haz_date}")
    print(f"  Total HAZARDOUS samples      : {total_hazardous}")
    print(f"  Distinct dates with HAZARDOUS: {n_distinct_dates}")

    # Sufficiency assessment
    print(f"\n  Sufficiency assessment:")
    if n_distinct_dates >= 2:
        print(f"  → HAZARDOUS spans {n_distinct_dates} distinct dates ({first_haz_date} to {last_haz_date}).")
        print(f"    A chronological split CAN place HAZARDOUS samples in both train and test,")
        print(f"    provided the cutoff falls between {first_haz_date} and {last_haz_date}.")
    else:
        print(f"  → HAZARDOUS only on 1 date. Any cutoff will put ALL HAZARDOUS in one partition.")
        print(f"    Chronological evaluation of HAZARDOUS recognition is NOT possible.")

# ── 11. Class distribution per calendar month ────────────────
print("\n" + "=" * 70)
print("  CLASS DISTRIBUTION PER CALENDAR MONTH")
print("=" * 70)

df["_month"] = df["_datetime"].dt.to_period("M")
monthly = df.groupby(["_month", "RiskLevel"]).size().unstack(fill_value=0)

# Ensure all classes are columns
for cls in ALL_CLASSES:
    if cls not in monthly.columns:
        monthly[cls] = 0
monthly = monthly[ALL_CLASSES]

print(f"\n  {'Month':<10}  " + "  ".join(f"{c[:6]:>8}" for c in ALL_CLASSES) + "   TOTAL")
print(f"  {'-'*10}  " + "  ".join(f"{'--------':>8}" for _ in ALL_CLASSES) + "   -----")
for month, row in monthly.iterrows():
    total = row.sum()
    print(f"  {str(month):<10}  " + "  ".join(f"{row[c]:>8}" for c in ALL_CLASSES) + f"   {total:>5}")

col_totals = monthly.sum()
print(f"  {'TOTAL':<10}  " + "  ".join(f"{col_totals[c]:>8}" for c in ALL_CLASSES) + f"   {col_totals.sum():>5}")

# ── 12. Candidate chronological splits ───────────────────────
print("\n" + "=" * 70)
print("  CANDIDATE CHRONOLOGICAL SPLIT ANALYSIS")
print("=" * 70)
print("  (whole calendar days — no date shared between train and test)")

all_dates = sorted(df["_date"].unique())
n_dates   = len(all_dates)
print(f"\n  Total distinct calendar dates in dataset: {n_dates}")
print(f"  Date range: {all_dates[0]}  →  {all_dates[-1]}\n")

# Evaluate a selection of cutoff positions
print(f"  {'Cutoff after':<14}  {'Train rows':>10}  {'Test rows':>9}  "
      f"{'Train HAZ':>9}  {'Test HAZ':>8}  {'Both HAZ?':>9}  {'Notes'}")
print(f"  {'-'*14}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*25}")

# Check cutoffs at every distinct date
for i in range(1, n_dates):
    cutoff_date = all_dates[i - 1]   # last date in training
    train_mask  = df["_date"] <= cutoff_date
    test_mask   = df["_date"] >  cutoff_date

    train_set = df[train_mask]
    test_set  = df[test_mask]

    train_rows = len(train_set)
    test_rows  = len(test_set)
    train_haz  = (train_set["RiskLevel"] == "HAZARDOUS").sum()
    test_haz   = (test_set["RiskLevel"]  == "HAZARDOUS").sum()
    both_haz   = "YES" if (train_haz > 0 and test_haz > 0) else "NO"

    # Only print rows where at least something interesting is happening
    # (always print if both_haz or at meaningful split fractions)
    pct_train = train_rows / len(df) * 100
    should_print = (
        both_haz == "YES" or
        abs(pct_train - 60) < 5 or
        abs(pct_train - 70) < 5 or
        abs(pct_train - 75) < 5 or
        abs(pct_train - 80) < 5 or
        i == 1 or
        i == n_dates - 1
    )
    if should_print:
        notes = []
        if both_haz == "YES":
            notes.append("HAZ in both")
        train_classes = set(train_set["RiskLevel"].unique())
        test_classes  = set(test_set["RiskLevel"].unique())
        if train_classes != test_classes:
            missing_in_test  = train_classes - test_classes
            missing_in_train = test_classes - train_classes
            if missing_in_test:
                notes.append(f"test missing: {','.join(sorted(missing_in_test))[:20]}")
            if missing_in_train:
                notes.append(f"train missing: {','.join(sorted(missing_in_train))[:20]}")
        note_str = "; ".join(notes) if notes else ""
        print(f"  {str(cutoff_date):<14}  {train_rows:>10}  {test_rows:>9}  "
              f"{train_haz:>9}  {test_haz:>8}  {both_haz:>9}  {note_str}")

print("\n" + "=" * 70)
print("  Analysis complete.")
print("=" * 70)
