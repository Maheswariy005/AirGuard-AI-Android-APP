# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd

cm = pd.read_csv(r"c:\Users\ADMIN\AppData\AirGuardAI\ml_pipeline\experiments\final_audit\confusion_matrix.csv", index_col=0)
print("Confusion matrix (from saved CSV):")
print(cm.to_string())
print()
print("Per-class metrics derived from confusion matrix:")
print(f"{'Class':<28} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Supp':>6}")
print("-"*75)
for cls in cm.index:
    tp = cm.loc[cls, cls]
    fp = cm[cls].sum() - tp
    fn = cm.loc[cls].sum() - tp
    sup = cm.loc[cls].sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    print(f"{cls:<28} {int(tp):>5} {int(fp):>5} {int(fn):>5} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {int(sup):>6}")
print()
total_correct = sum(cm.loc[c,c] for c in cm.index)
total = cm.values.sum()
print(f"Overall accuracy: {total_correct}/{total} = {total_correct/total:.6f} ({total_correct/total*100:.4f}%)")
