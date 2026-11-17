# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\ADMIN\Downloads\AeroSense Device 2 - Sheet1 (1).csv"
ALL_CLASSES = ['GOOD','MODERATE','UNHEALTHY_SENSITIVE','UNHEALTHY','VERY_UNHEALTHY','HAZARDOUS']

def aqi_to_risk(aqi):
    if   aqi <= 20: return "GOOD"
    elif aqi <= 40: return "MODERATE"
    elif aqi <= 60: return "UNHEALTHY_SENSITIVE"
    elif aqi <= 80: return "UNHEALTHY"
    elif aqi <= 90: return "VERY_UNHEALTHY"
    else:           return "HAZARDOUS"

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df = df.replace("N/A", np.nan)
feat_cols = [c for c in df.columns if c not in ['Date','Time','AQI']]
for col in feat_cols + ['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates()
df = df.dropna(subset=feat_cols + ['AQI'])
df['RiskLevel'] = df['AQI'].apply(aqi_to_risk)
df['_datetime'] = pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str), errors='coerce')
df = df.dropna(subset=['_datetime'])
df = df.sort_values('_datetime').reset_index(drop=True)
df['_date'] = df['_datetime'].dt.date

print(f"Total usable rows: {len(df)}")
print(f"Date range: {df['_date'].min()} to {df['_date'].max()}")

# Candidate cutoffs to evaluate
import datetime
candidates = [
    datetime.date(2026, 4, 1),
    datetime.date(2026, 4, 15),
    datetime.date(2026, 5, 1),
    datetime.date(2026, 5, 15),
    datetime.date(2026, 5, 29),
    datetime.date(2026, 6, 1),
    datetime.date(2026, 7, 1),
    datetime.date(2026, 7, 14),  # day BEFORE second hazardous date
]

results = []
print("\n" + "="*80)
for cutoff in candidates:
    tr = df[df['_date'] < cutoff]
    te = df[df['_date'] >= cutoff]
    if len(tr) == 0 or len(te) == 0:
        print(f"Cutoff {cutoff}: SKIPPED (empty split)")
        continue

    tr_haz = (tr['RiskLevel']=='HAZARDOUS').sum()
    te_haz = (te['RiskLevel']=='HAZARDOUS').sum()

    tr_classes = {c: (tr['RiskLevel']==c).sum() for c in ALL_CLASSES}
    te_classes = {c: (te['RiskLevel']==c).sum() for c in ALL_CLASSES}

    tr_present = sum(1 for v in tr_classes.values() if v > 0)
    te_present = sum(1 for v in te_classes.values() if v > 0)

    haz_in_both = tr_haz >= 30 and te_haz >= 30
    viable = tr_present >= 3 and te_present >= 3 and haz_in_both

    # check no date overlap
    tr_dates = set(tr['_date'].unique())
    te_dates = set(te['_date'].unique())
    overlap = tr_dates & te_dates
    
    pct_tr = len(tr)/len(df)*100

    print(f"\nCutoff: {cutoff}  (Train {pct_tr:.1f}% / Test {100-pct_tr:.1f}%)")
    print(f"  Train: {len(tr):>6} rows  {tr['_date'].min()} -> {tr['_date'].max()}")
    print(f"  Test : {len(te):>6} rows  {te['_date'].min()} -> {te['_date'].max()}")
    print(f"  Date overlap between sets: {len(overlap)} dates  {'[CLEAN]' if len(overlap)==0 else '[PROBLEM]'}")
    print(f"  Train class counts:")
    for c in ALL_CLASSES:
        flag = " <-- ZERO" if tr_classes[c]==0 else ""
        print(f"    {c:<26}: {tr_classes[c]:>6}{flag}")
    print(f"  Test class counts:")
    for c in ALL_CLASSES:
        flag = " <-- ZERO" if te_classes[c]==0 else ""
        print(f"    {c:<26}: {te_classes[c]:>6}{flag}")
    print(f"  HAZARDOUS in train: {tr_haz}  |  HAZARDOUS in test: {te_haz}")
    print(f"  HAZARDOUS in both (>=30 each): {'YES' if haz_in_both else 'NO'}")
    print(f"  VIABLE: {'YES' if viable else 'NO'}")
    
    results.append({
        'cutoff': str(cutoff),
        'train_n': len(tr), 'test_n': len(te),
        'pct_train': pct_tr,
        'tr_haz': tr_haz, 'te_haz': te_haz,
        'haz_both': haz_in_both, 'viable': viable,
        'tr_present': tr_present, 'te_present': te_present,
        'overlap': len(overlap)
    })

print("\n" + "="*80)
print("\nSUMMARY TABLE:")
print(f"{'Cutoff':<14} {'Train':>7} {'Test':>7} {'TrHAZ':>7} {'TeHAZ':>7} {'HAZboth':>8} {'Viable':>7}")
print("-"*58)
for r in results:
    print(f"{r['cutoff']:<14} {r['train_n']:>7} {r['test_n']:>7} {r['tr_haz']:>7} {r['te_haz']:>7} {'YES' if r['haz_both'] else 'NO':>8} {'YES' if r['viable'] else 'NO':>7}")

# Determine best cutoff
viable_cuts = [r for r in results if r['viable']]
if viable_cuts:
    # pick the one with most training data (earliest cutoff) among viable ones
    best = min(viable_cuts, key=lambda r: r['pct_train'] if r['pct_train'] < 100 else 999)
    # actually pick the one that maximises train data while keeping >=30 haz in test
    best = max(viable_cuts, key=lambda r: r['train_n'])
    print(f"\nBEST VIABLE CUTOFF: {best['cutoff']}")
    print(f"  Train: {best['train_n']} rows  Test: {best['test_n']} rows")
    print(f"  HAZARDOUS train: {best['tr_haz']}  test: {best['te_haz']}")
else:
    # find cutoff that maximises haz in training with at least 50 in test
    with_haz_test = [r for r in results if r['te_haz'] >= 50]
    if with_haz_test:
        best = max(with_haz_test, key=lambda r: r['tr_haz'])
        print(f"\nNO FULLY VIABLE CUTOFF FOUND.")
        print(f"Best partial cutoff (max train HAZ with >=50 test HAZ): {best['cutoff']}")
        print(f"  Train HAZ: {best['tr_haz']}  Test HAZ: {best['te_haz']}")
    else:
        print("\nNO VIABLE CUTOFF: cannot put >=30 HAZARDOUS in both train and test.")
        print("HAZARDOUS is only present on 4 dates (all from 2026-05-30 onward).")
        print("ANY chronological cutoff before 2026-05-30 gives 0 HAZARDOUS in training.")
        print("The dataset does not support a chronological split that trains on HAZARDOUS.")
