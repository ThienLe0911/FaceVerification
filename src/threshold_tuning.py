#!/usr/bin/env python3
"""
threshold_tuning.py - Find Optimal Threshold for Face Verification

This script implements DAY 7 of the timeline (github/timeline.txt):
"Find Optimal Threshold"

Purpose:
- Load positive and negative similarity scores from DAY 6
- Compute ROC curve, AUC, and various optimal threshold methods
- Save comprehensive threshold analysis results

Methods for threshold optimization:
1. Youden's J statistic (max TPR - FPR)
2. Equal Error Rate (EER) - where FPR = FNR  
3. Maximum F1-score
4. Minimum total error rate

Example usage:
    python src/threshold_tuning.py --pos data/evaluation/similarities_pos.npy \
                                   --neg data/evaluation/similarities_neg.npy \
                                   --out_dir results/

Output:
    results/threshold_results.json - Complete analysis results
    Console output with detailed statistics and recommendations
"""

import argparse
import json
import logging
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

try:
    from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
except ImportError:
    print("❌ Error: scikit-learn not installed. Please install with: pip install scikit-learn")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_similarity_data(pos_file: str, neg_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load positive and negative similarity scores.
    
    Args:
        pos_file: Path to positive similarities .npy file
        neg_file: Path to negative similarities .npy file
        
    Returns:
        Tuple of (y_true, y_score) arrays for ROC analysis
    """
    logger.info(f"Loading positive similarities from: {pos_file}")
    pos_similarities = np.load(pos_file)
    
    logger.info(f"Loading negative similarities from: {neg_file}")
    neg_similarities = np.load(neg_file)
    
    logger.info(f"Loaded {len(pos_similarities)} positive and {len(neg_similarities)} negative similarities")
    
    # Create y_true and y_score arrays
    y_true = np.concatenate([
        np.ones(len(pos_similarities)),    # Positive class = 1
        np.zeros(len(neg_similarities))    # Negative class = 0
    ])
    
    y_score = np.concatenate([pos_similarities, neg_similarities])
    
    return y_true, y_score


def compute_roc_analysis(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (similarity values)
        
    Returns:
        Dictionary with ROC analysis results
    """
    logger.info("Computing ROC curve and AUC...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(), 
        'thresholds': thresholds.tolist(),
        'auc': roc_auc
    }


def find_youden_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Dict:
    """
    Find optimal threshold using Youden's J statistic.
    J = TPR - FPR = Sensitivity + Specificity - 1
    
    Args:
        fpr: False positive rates
        tpr: True positive rates  
        thresholds: Corresponding thresholds
        
    Returns:
        Dictionary with Youden threshold results
    """
    j_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(j_scores)
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_j = j_scores[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    return {
        'method': 'Youden J Statistic',
        'threshold': float(optimal_threshold),
        'j_score': float(optimal_j),
        'tpr': float(optimal_tpr),
        'fpr': float(optimal_fpr),
        'specificity': float(1 - optimal_fpr),
        'sensitivity': float(optimal_tpr)
    }


def find_eer_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Dict:
    """
    Find Equal Error Rate (EER) threshold where FPR = FNR.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Corresponding thresholds
        
    Returns:
        Dictionary with EER threshold results  
    """
    fnr = 1 - tpr  # False Negative Rate = 1 - TPR
    
    # Find point where |FPR - FNR| is minimized
    eer_diff = np.abs(fpr - fnr)
    eer_idx = np.argmin(eer_diff)
    
    eer_threshold = thresholds[eer_idx]
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2  # Average of FPR and FNR
    
    return {
        'method': 'Equal Error Rate (EER)',
        'threshold': float(eer_threshold),
        'eer_rate': float(eer_rate),
        'fpr': float(fpr[eer_idx]),
        'fnr': float(fnr[eer_idx]),
        'tpr': float(tpr[eer_idx])
    }


def find_max_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Find threshold that maximizes F1-score.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Dictionary with max F1 threshold results
    """
    # Use precision-recall curve to find range of thresholds
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    
    # Calculate F1-score for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Get corresponding precision and recall
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    
    return {
        'method': 'Maximum F1-Score',
        'threshold': float(optimal_threshold),
        'f1_score': float(optimal_f1),
        'precision': float(optimal_precision),
        'recall': float(optimal_recall)
    }


def find_min_error_threshold(y_true: np.ndarray, y_score: np.ndarray, 
                           fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Dict:
    """
    Find threshold that minimizes total error rate.
    Total Error = FPR * P(negative) + FNR * P(positive)
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        fpr: False positive rates
        tpr: True positive rates  
        thresholds: Corresponding thresholds
        
    Returns:
        Dictionary with minimum error threshold results
    """
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    total_samples = len(y_true)
    
    p_positive = n_positive / total_samples
    p_negative = n_negative / total_samples
    
    fnr = 1 - tpr  # False Negative Rate
    
    # Total error rate for each threshold
    total_error = fpr * p_negative + fnr * p_positive
    
    optimal_idx = np.argmin(total_error)
    optimal_threshold = thresholds[optimal_idx]
    optimal_error = total_error[optimal_idx]
    
    return {
        'method': 'Minimum Total Error',
        'threshold': float(optimal_threshold),
        'total_error_rate': float(optimal_error),
        'fpr': float(fpr[optimal_idx]),
        'fnr': float(fnr[optimal_idx]),
        'tpr': float(tpr[optimal_idx]),
        'p_positive': float(p_positive),
        'p_negative': float(p_negative)
    }


def compute_threshold_statistics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    """
    Compute classification statistics for a given threshold.
    
    Args:
        y_true: True binary labels
        y_score: Target scores  
        threshold: Classification threshold
        
    Returns:
        Dictionary with classification metrics
    """
    y_pred = (y_score >= threshold).astype(int)
    
    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    tpr = recall  # Same as recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = 1 - fpr
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'specificity': float(specificity),
        'confusion_matrix': {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        }
    }


def save_results(results: Dict, output_dir: str):
    """
    Save threshold analysis results to JSON file.
    
    Args:
        results: Complete threshold analysis results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'threshold_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {output_file}")


def print_summary(results: Dict):
    """
    Print formatted summary of threshold analysis results.
    
    Args:
        results: Complete threshold analysis results
    """
    print("\n" + "=" * 80)
    print("🎯 THRESHOLD TUNING ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Positive samples: {results['dataset_info']['n_positive']}")
    print(f"   Negative samples: {results['dataset_info']['n_negative']}")
    print(f"   Total samples: {results['dataset_info']['total_samples']}")
    print(f"   ROC AUC: {results['roc_analysis']['auc']:.4f}")
    
    print(f"\n🎯 Optimal Threshold Methods:")
    print("-" * 50)
    
    methods = ['youden', 'eer', 'max_f1', 'min_error']
    method_names = ['Youden J', 'EER', 'Max F1', 'Min Error']
    
    for method, name in zip(methods, method_names):
        thresh_data = results['threshold_methods'][method]
        print(f"\n📌 {name}:")
        print(f"   Threshold: {thresh_data['threshold']:.4f}")
        
        if 'j_score' in thresh_data:
            print(f"   J-score: {thresh_data['j_score']:.4f}")
        if 'eer_rate' in thresh_data:
            print(f"   EER rate: {thresh_data['eer_rate']:.4f}")
        if 'f1_score' in thresh_data:
            print(f"   F1-score: {thresh_data['f1_score']:.4f}")
        if 'total_error_rate' in thresh_data:
            print(f"   Total error: {thresh_data['total_error_rate']:.4f}")
    
    # Recommendation
    print(f"\n🏆 RECOMMENDED THRESHOLD:")
    print(f"   Based on Youden J-statistic: {results['threshold_methods']['youden']['threshold']:.4f}")
    print(f"   Expected Performance:")
    
    youden_stats = results['threshold_statistics']['youden']
    print(f"   • Accuracy: {youden_stats['accuracy']:.1%}")
    print(f"   • Precision: {youden_stats['precision']:.1%}")
    print(f"   • Recall: {youden_stats['recall']:.1%}")
    print(f"   • F1-score: {youden_stats['f1_score']:.1%}")
    print(f"   • FPR: {youden_stats['fpr']:.1%}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Face Verification Threshold Tuning')
    parser.add_argument('--pos', required=True, 
                       help='Path to positive similarities .npy file')
    parser.add_argument('--neg', required=True,
                       help='Path to negative similarities .npy file') 
    parser.add_argument('--out_dir', default='results/',
                       help='Output directory for results (default: results/)')
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting threshold tuning analysis...")
    logger.info(f"Positive similarities: {args.pos}")
    logger.info(f"Negative similarities: {args.neg}")
    logger.info(f"Output directory: {args.out_dir}")
    
    # Load data
    y_true, y_score = load_similarity_data(args.pos, args.neg)
    
    # Basic dataset info
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    # ROC analysis
    roc_analysis = compute_roc_analysis(y_true, y_score)
    fpr = np.array(roc_analysis['fpr'])
    tpr = np.array(roc_analysis['tpr'])
    thresholds = np.array(roc_analysis['thresholds'])
    
    # Find optimal thresholds using different methods
    logger.info("Computing optimal thresholds...")
    
    youden_result = find_youden_threshold(fpr, tpr, thresholds)
    eer_result = find_eer_threshold(fpr, tpr, thresholds) 
    f1_result = find_max_f1_threshold(y_true, y_score)
    error_result = find_min_error_threshold(y_true, y_score, fpr, tpr, thresholds)
    
    # Compute statistics for each optimal threshold
    threshold_stats = {}
    for name, result in [('youden', youden_result), ('eer', eer_result), 
                        ('max_f1', f1_result), ('min_error', error_result)]:
        threshold_stats[name] = compute_threshold_statistics(y_true, y_score, result['threshold'])
    
    # Compile complete results
    results = {
        'timestamp': datetime.now().isoformat(),
        'input_files': {
            'positive_similarities': args.pos,
            'negative_similarities': args.neg
        },
        'dataset_info': {
            'n_positive': int(n_positive),
            'n_negative': int(n_negative), 
            'total_samples': int(len(y_true)),
            'positive_ratio': float(n_positive / len(y_true))
        },
        'roc_analysis': roc_analysis,
        'threshold_methods': {
            'youden': youden_result,
            'eer': eer_result,
            'max_f1': f1_result,
            'min_error': error_result
        },
        'threshold_statistics': threshold_stats
    }
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.out_dir)
    
    logger.info("🎉 Threshold tuning analysis completed successfully!")
    logger.info(f"📊 ROC AUC: {roc_analysis['auc']:.4f}")
    logger.info(f"🏆 Recommended threshold (Youden): {youden_result['threshold']:.4f}")


if __name__ == "__main__":
    main()