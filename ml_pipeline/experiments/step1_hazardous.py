# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
FEATURES = ["Temperature(deg C)","Humidity(%)","Pressure(hPa)","Gas (kOhms)","Altitude(m)","DewPoint(deg C)","HeatIndex(deg C)"]

def aqi_to_risk(aqi):
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df.replace("N/A", np.nan, inplace=True)

# get actual column names
print("Columns:", list(df.columns))

feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
print("Feature columns found:", feat_cols)

for col in feat_cols + ['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

n0 = len(df)
df.drop_duplicates(inplace=True)
n1 = len(df)
df.dropna(subset=feat_cols + ['AQI'], inplace=True)
n2 = len(df)
print(f"Raw: {n0}  After dedup: {n1}  After dropna: {n2}")

df['RiskLevel'] = df['AQI'].apply(aqi_to_risk)
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df.dropna(subset=['_datetime'], inplace=True)
df.sort_values('_datetime', inplace=True)
df['_date'] = df['_datetime'].dt.date
print(f"Rows with valid datetime: {len(df)}")

# VERY_UNHEALTHY total
vu = (df['RiskLevel']=='VERY_UNHEALTHY').sum()
print(f"\nVERY_UNHEALTHY total samples: {vu}")
print("AQI values in 81-90 range:", df[(df['AQI']>80) & (df['AQI']<=90)]['AQI'].unique().tolist())

# HAZARDOUS by date
haz = df[df['RiskLevel']=='HAZARDOUS']
print(f"\nTotal HAZARDOUS samples: {len(haz)}")
print(f"First HAZARDOUS date: {haz['_datetime'].min()}")
print(f"Last  HAZARDOUS date: {haz['_datetime'].max()}")
haz_dates = haz.groupby('_date')
distinct = haz_dates.ngroups
print(f"Distinct dates with HAZARDOUS: {distinct}")

print("\nHAZARDOUS by date:")
print(f"{'Date':<12} {'Total':>7} {'HAZARDOUS':>10} {'HAZ%':>7} {'AQI_min':>8} {'AQI_max':>8}")
print("-"*58)
for d, grp in haz_dates:
    total_day = len(df[df['_date']==d])
    h_cnt = len(grp)
    pct = h_cnt/total_day*100
    print(f"{str(d):<12} {total_day:>7} {h_cnt:>10} {pct:>7.1f}% {grp['AQI'].min():>8.0f} {grp['AQI'].max():>8.0f}")

# Monthly class distribution
print("\nMonthly class distribution:")
df['_ym'] = df['_datetime'].dt.to_period('M')
monthly = df.groupby('_ym')['RiskLevel'].value_counts().unstack(fill_value=0)
for col in ['GOOD','MODERATE','UNHEALTHY_SENSITIVE','UNHEALTHY','VERY_UNHEALTHY','HAZARDOUS']:
    if col not in monthly.columns:
        monthly[col] = 0
monthly = monthly[['GOOD','MODERATE','UNHEALTHY_SENSITIVE','UNHEALTHY','VERY_UNHEALTHY','HAZARDOUS']]
print(monthly.to_string())

# Overall class distribution
print("\nOverall class distribution:")
for cls in ['GOOD','MODERATE','UNHEALTHY_SENSITIVE','UNHEALTHY','VERY_UNHEALTHY','HAZARDOUS']:
    cnt = (df['RiskLevel']==cls).sum()
    print(f"  {cls:<26}: {cnt:>6}  ({cnt/len(df)*100:.2f}%)")
