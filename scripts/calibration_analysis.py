#!/usr/bin/env python3
"""
Analyze calibration metrics for NLI model predictions.
Computes reliability table, ECE, NLL, and Brier Score.
Supports both TSV error files and JSONL full prediction files.
"""

import csv
import json
import numpy as np
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple
import scipy.special


def load_predictions_jsonl(jsonl_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions from JSONL file.
    
    Returns:
        confidences: Array of confidence scores (max probability)
        predictions: Array of predicted labels (0, 1, or 2)
        labels: Array of true labels (0, 1, or 2)
        probs: Array of shape (n_samples, 3) with probabilities for each class
    """
    confidences = []
    predictions = []
    labels = []
    probs_list = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Get predicted scores (logits)
            scores = np.array(data['predicted_scores'])
            
            # Convert logits to probabilities using softmax
            probs = scipy.special.softmax(scores)
            probs_list.append(probs)
            
            # Get confidence (max probability)
            conf = np.max(probs)
            confidences.append(conf)
            
            # Get predicted label (already provided)
            pred_label = data['predicted_label']
            predictions.append(pred_label)
            
            # Get true label
            true_label = data['label']
            labels.append(true_label)
    
    return (np.array(confidences), np.array(predictions), 
            np.array(labels), np.array(probs_list))


def load_predictions_tsv(tsv_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions from TSV file.
    
    Returns:
        confidences: Array of confidence scores (max probability)
        predictions: Array of predicted labels (0, 1, or 2)
        labels: Array of true labels (0, 1, or 2)
    """
    confidences = []
    predictions = []
    labels = []
    
    label_map = {
        'entailment': 0,
        'neutral': 1, 
        'contradiction': 2
    }
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Get confidence (max probability)
            conf = float(row['confidence'])
            confidences.append(conf)
            
            # Get predicted and true labels
            pred_label = label_map[row['predicted_label']]
            true_label = label_map[row['true_label']]
            
            predictions.append(pred_label)
            labels.append(true_label)
    
    return np.array(confidences), np.array(predictions), np.array(labels)


def load_full_probabilities_tsv(tsv_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load full probability distributions from TSV file.
    
    Returns:
        probs: Array of shape (n_samples, 3) with probabilities for each class
        labels: Array of true labels (0, 1, or 2)
    """
    probs_list = []
    labels = []
    
    label_map = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Get full probability distribution
            entail_prob = float(row['entailment_prob'])
            neutral_prob = float(row['neutral_prob'])
            contra_prob = float(row['contradiction_prob'])
            
            probs_list.append([entail_prob, neutral_prob, contra_prob])
            
            # Get true label
            true_label = label_map[row['true_label']]
            labels.append(true_label)
    
    return np.array(probs_list), np.array(labels)


def compute_reliability_table(confidences: np.ndarray, predictions: np.ndarray, 
                             labels: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute reliability table with binned confidence vs accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    reliability_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find examples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            # Calculate accuracy in this bin
            bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_count = np.sum(in_bin)
            
            reliability_data.append({
                'bin_range': f"({bin_lower:.2f}, {bin_upper:.2f}]",
                'bin_center': (bin_lower + bin_upper) / 2,
                'count': int(bin_count),
                'avg_confidence': float(bin_confidence),
                'accuracy': float(bin_accuracy),
                'gap': float(bin_confidence - bin_accuracy)
            })
        else:
            reliability_data.append({
                'bin_range': f"({bin_lower:.2f}, {bin_upper:.2f}]",
                'bin_center': (bin_lower + bin_upper) / 2,
                'count': 0,
                'avg_confidence': 0,
                'accuracy': 0,
                'gap': 0
            })
    
    return reliability_data


def compute_mce(confidences: np.ndarray, predictions: np.ndarray, 
                labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE = max_b |accuracy_b - confidence_b|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return float(mce)


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, 
                labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = Σ (n_b / N) * |accuracy_b - confidence_b|
    where b indexes bins, n_b is the number of samples in bin b, and N is total samples.
    """
    n = len(confidences)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.sum(in_bin) / n
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return float(ece)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Negative Log-Likelihood.
    
    NLL = -1/N * Σ log(p_i[y_i])
    where p_i[y_i] is the predicted probability for the true class.
    """
    n = len(labels)
    # Get probability of true class for each example
    true_probs = probs[np.arange(n), labels]
    
    # Clip to avoid log(0)
    true_probs = np.clip(true_probs, 1e-7, 1)
    
    nll = -np.mean(np.log(true_probs))
    return float(nll)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier Score for multi-class classification.
    
    BS = 1/N * Σ Σ (p_ij - y_ij)^2
    where p_ij is predicted probability for class j of sample i,
    and y_ij is 1 if sample i belongs to class j, else 0.
    """
    n_samples = len(labels)
    n_classes = probs.shape[1]
    
    # Convert labels to one-hot encoding
    labels_onehot = np.zeros((n_samples, n_classes))
    labels_onehot[np.arange(n_samples), labels] = 1
    
    # Compute Brier score
    brier = np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))
    return float(brier)


def print_reliability_diagram(reliability_data: List[Dict]):
    """
    Print an ASCII reliability diagram.
    """
    print("\n" + "="*70)
    print("RELIABILITY DIAGRAM")
    print("="*70)
    print("\nPerfect calibration: confidence = accuracy (diagonal line)")
    print("Below diagonal: overconfident | Above diagonal: underconfident\n")
    
    # Create ASCII plot (20x10 grid)
    height = 10
    width = 40
    
    # Initialize grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Add axes
    for i in range(height):
        grid[i][0] = '│'
    for j in range(width):
        grid[height-1][j] = '─'
    grid[height-1][0] = '└'
    
    # Add diagonal line (perfect calibration)
    for i in range(min(height, width)):
        y = height - 1 - int(i * height / width)
        x = i
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = '·'
    
    # Plot reliability data points
    for bin_data in reliability_data:
        if bin_data['count'] > 0:
            x = int(bin_data['avg_confidence'] * (width - 1))
            y = height - 1 - int(bin_data['accuracy'] * height)
            
            if 0 <= y < height and 0 <= x < width:
                # Use different symbols based on sample count
                if bin_data['count'] > 1000:
                    symbol = '█'
                elif bin_data['count'] > 500:
                    symbol = '●'
                elif bin_data['count'] > 100:
                    symbol = 'o'
                else:
                    symbol = '.'
                grid[y][x] = symbol
    
    # Print grid with labels
    print("  1.0 ┤")
    for i in range(height):
        acc = 1.0 - (i / height)
        print(f"  {acc:.1f} │{''.join(grid[i])}")
    print("      └" + "─"*width)
    print("        0.0" + " "*15 + "0.5" + " "*15 + "1.0")
    print("                      Confidence →")
    print("      ↑ Accuracy")
    
    # Legend
    print("\nSymbols: █ >1000 | ● >500 | o >100 | . ≤100 | · perfect")


def print_calibration_table(reliability_data: List[Dict]):
    """
    Print a formatted calibration table.
    """
    print("\n" + "="*70)
    print("CALIBRATION TABLE")
    print("="*70)
    print(f"{'Confidence Bin':^15} | {'Count':^8} | {'Avg Conf':^10} | {'Accuracy':^10} | {'Gap':^10}")
    print("-"*70)
    
    for bin_data in reliability_data:
        if bin_data['count'] > 0:
            print(f"{bin_data['bin_range']:^15} | {bin_data['count']:^8} | "
                  f"{bin_data['avg_confidence']:^10.3f} | {bin_data['accuracy']:^10.3f} | "
                  f"{bin_data['gap']:+10.3f}")
        else:
            print(f"{bin_data['bin_range']:^15} | {'-':^8} | {'-':^10} | {'-':^10} | {'-':^10}")


def main():
    parser = argparse.ArgumentParser(description='Compute calibration metrics for NLI model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file with predictions (TSV or JSONL)')
    parser.add_argument('--n-bins', type=int, default=10,
                        help='Number of bins for calibration analysis')
    
    args = parser.parse_args()
    
    # Load predictions based on file type
    print(f"Loading predictions from {args.input}...")
    
    if args.input.endswith('.jsonl'):
        # Full predictions file
        confidences, predictions, labels, probs = load_predictions_jsonl(args.input)
        true_labels = labels
    else:
        # TSV error file
        confidences, predictions, labels = load_predictions_tsv(args.input)
        probs, true_labels = load_full_probabilities_tsv(args.input)
    
    # Basic statistics
    n_samples = len(confidences)
    accuracy = np.mean(predictions == labels)
    avg_confidence = np.mean(confidences)
    
    # Per-class accuracy if using full predictions
    unique_labels = np.unique(labels)
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    
    print("\n" + "="*70)
    print("BASIC STATISTICS")
    print("="*70)
    print(f"Total samples: {n_samples:,}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Confidence - Accuracy gap: {avg_confidence - accuracy:+.4f}")
    
    if args.input.endswith('.jsonl'):
        print("\nPer-class accuracy:")
        for label in unique_labels:
            mask = labels == label
            class_acc = np.mean(predictions[mask] == labels[mask])
            class_count = np.sum(mask)
            print(f"  {class_names[label]:15}: {class_acc:.4f} ({class_count:,} samples)")
    
    # Compute reliability table
    reliability_data = compute_reliability_table(confidences, predictions, labels, args.n_bins)
    
    # Print reliability diagram
    print_reliability_diagram(reliability_data)
    
    # Print calibration table
    print_calibration_table(reliability_data)
    
    # Compute calibration metrics
    ece = compute_ece(confidences, predictions, labels, args.n_bins)
    mce = compute_mce(confidences, predictions, labels, args.n_bins)
    nll = compute_nll(probs, true_labels)
    brier = compute_brier_score(probs, true_labels)
    
    print("\n" + "="*70)
    print("CALIBRATION METRICS")
    print("="*70)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  → Lower is better (0 = perfect calibration)")
    print(f"  → Weighted average gap between confidence and accuracy")
    
    print(f"\nMaximum Calibration Error (MCE): {mce:.4f}")
    print(f"  → Lower is better (0 = perfect calibration)")
    print(f"  → Maximum gap between confidence and accuracy across bins")
    
    print(f"\nNegative Log-Likelihood (NLL): {nll:.4f}")
    print(f"  → Lower is better")
    print(f"  → Measures quality of probability estimates")
    
    print(f"\nBrier Score: {brier:.4f}")
    print(f"  → Lower is better (0 = perfect, 2 = worst)")
    print(f"  → Measures accuracy of probabilistic predictions")
    
    # Confidence distribution
    print("\n" + "="*70)
    print("CONFIDENCE DISTRIBUTION")
    print("="*70)
    
    conf_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    print(f"{'Range':^15} | {'Count':^10} | {'Percentage':^12} | {'Accuracy':^10}")
    print("-"*60)
    
    for i in range(len(conf_bins)):
        if i == 0:
            mask = confidences <= conf_bins[i]
            range_str = f"[0.00, {conf_bins[i]:.2f}]"
        else:
            mask = (confidences > conf_bins[i-1]) & (confidences <= conf_bins[i])
            range_str = f"({conf_bins[i-1]:.2f}, {conf_bins[i]:.2f}]"
        
        count = np.sum(mask)
        pct = 100 * count / n_samples
        
        if count > 0:
            acc = np.mean(predictions[mask] == labels[mask])
            print(f"{range_str:^15} | {count:^10,} | {pct:^11.1f}% | {acc:^10.3f}")
        else:
            print(f"{range_str:^15} | {count:^10} | {pct:^11.1f}% | {'-':^10}")
    
    # Error analysis by confidence (only for full predictions)
    if args.input.endswith('.jsonl'):
        print("\n" + "="*70)
        print("ERROR ANALYSIS BY CONFIDENCE")
        print("="*70)
        
        errors_mask = predictions != labels
        n_errors = np.sum(errors_mask)
        print(f"Total errors: {n_errors:,} ({100*n_errors/n_samples:.1f}%)")
        
        print("\nError distribution by confidence:")
        for i in range(len(conf_bins)):
            if i == 0:
                mask = (confidences <= conf_bins[i]) & errors_mask
                range_str = f"[0.00, {conf_bins[i]:.2f}]"
            else:
                mask = (confidences > conf_bins[i-1]) & (confidences <= conf_bins[i]) & errors_mask
                range_str = f"({conf_bins[i-1]:.2f}, {conf_bins[i]:.2f}]"
            
            count = np.sum(mask)
            pct_of_errors = 100 * count / n_errors if n_errors > 0 else 0
            
            if count > 0:
                avg_conf = np.mean(confidences[mask])
                print(f"{range_str:^15} | {count:^6,} errors | {pct_of_errors:^11.1f}% of errors | avg conf: {avg_conf:.3f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()