#!/usr/bin/env python3
"""
threshold_tuning_v2.py - Find optimal verification threshold (Alternative approach)

Purpose:
- Load similarity scores (positive & negative)
- Compute ROC, AUC
- Find optimal threshold using:
    * Youden's J statistic
    * Equal Error Rate (EER)
    * Maximum F1-score
    * Minimum total error
- Recommended threshold = (Youden + Max F1) / 2

Timeline Reference: DAY 7 — Threshold Tuning (Alternative Implementation)
"""

import os
import argparse
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_curve, auc, f1_score

def compute_eer(fpr, fnr, thresholds):
    """Find threshold where FPR ≈ FNR."""
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    return thresholds[idx], fpr[idx]

def compute_best_f1(pos_scores, neg_scores):
    """Compute threshold that maximizes F1-score."""
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    thresholds = np.linspace(-1, 1, 2000)
    best_f1 = 0
    best_t = 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

def compute_min_total_error(fpr, fnr, thresholds):
    """Find threshold with FPR + FNR minimized."""
    total_error = fpr + fnr
    idx = np.argmin(total_error)
    return thresholds[idx], total_error[idx]

def main():
    parser = argparse.ArgumentParser(description="Threshold tuning v2")
    parser.add_argument("--pos", required=True, help="Path to positive similarities (npy)")
    parser.add_argument("--neg", required=True, help="Path to negative similarities (npy)")
    parser.add_argument("--out_dir", default="data/evaluation/threshold", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("🎯 THRESHOLD TUNING v2 - Alternative Approach")
    print("=" * 60)

    # Load similarity scores
    pos_scores = np.load(args.pos)
    neg_scores = np.load(args.neg)

    print("📊 Data Loaded:")
    print(f"  Positive samples: {len(pos_scores)}")
    print(f"  Negative samples: {len(neg_scores)}")
    print(f"  Positive range: [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")
    print(f"  Negative range: [{neg_scores.min():.4f}, {neg_scores.max():.4f}]")

    # Prepare labels and scores
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])

    # ROC curve
    print(f"\n🔍 Computing ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    auc_score = auc(fpr, tpr)

    print(f"ROC AUC = {auc_score:.4f}")

    print(f"\n📈 Finding Optimal Thresholds:")
    print("-" * 40)

    # 1) Youden's J statistic
    J = tpr - fpr
    j_idx = np.argmax(J)
    th_j = thresholds[j_idx]
    j_score = J[j_idx]
    print(f"1. Youden's J: {th_j:.4f} (J={j_score:.4f})")

    # 2) Equal Error Rate
    th_eer, eer_value = compute_eer(fpr, fnr, thresholds)
    print(f"2. EER: {th_eer:.4f} (EER={eer_value:.4f})")

    # 3) Maximum F1 score
    th_f1, best_f1 = compute_best_f1(pos_scores, neg_scores)
    print(f"3. Max F1: {th_f1:.4f} (F1={best_f1:.4f})")

    # 4) Minimum total error
    th_min_err, total_err = compute_min_total_error(fpr, fnr, thresholds)
    print(f"4. Min Total Error: {th_min_err:.4f} (error={total_err:.4f})")

    # Select recommended threshold:
    # Typically, Youden or Max F1 are best
    recommended = float((th_j + th_f1) / 2)
    
    print(f"\n🏆 RECOMMENDATION:")
    print(f"   Youden J:     {th_j:.4f}")
    print(f"   Max F1:       {th_f1:.4f}")
    print(f"   Average:      {recommended:.4f}")
    print(f"")
    print(f"🔥 RECOMMENDED THRESHOLD = {recommended:.4f}")

    # Save results
    results = {
        "timestamp": str(np.datetime64('now')),
        "input_files": {
            "positive": args.pos,
            "negative": args.neg
        },
        "dataset_stats": {
            "n_positive": int(len(pos_scores)),
            "n_negative": int(len(neg_scores)),
            "pos_range": [float(pos_scores.min()), float(pos_scores.max())],
            "neg_range": [float(neg_scores.min()), float(neg_scores.max())]
        },
        "metrics": {
            "auc": float(auc_score),
            "threshold_youden": float(th_j),
            "threshold_eer": float(th_eer),
            "threshold_f1": float(th_f1),
            "threshold_min_error": float(th_min_err),
            "recommended": recommended
        },
        "method": "average_of_youden_and_f1"
    }

    output_file = Path(args.out_dir) / "threshold_results_v2.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}")
    print("🎉 Threshold tuning v2 completed!")
    
    # Quick comparison with range analysis
    gap = pos_scores.min() - neg_scores.max()
    print(f"\n📊 Separation Analysis:")
    print(f"   Positive min: {pos_scores.min():.4f}")
    print(f"   Negative max: {neg_scores.max():.4f}")
    print(f"   Gap: {gap:.4f}")
    
    if gap > 0:
        print(f"   ✅ Perfect separation exists!")
        print(f"   💡 Any threshold in [{neg_scores.max():.4f}, {pos_scores.min():.4f}] works perfectly")
    else:
        print(f"   ⚠️ Overlap exists, threshold optimization needed")

if __name__ == "__main__":
    main()