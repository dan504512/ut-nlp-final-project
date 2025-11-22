#!/usr/bin/env python3
"""
Analyze errors from NLI model predictions on contrast sets and ANLI.
Generates a TSV file with error examples including confidence scores.
"""

import json
import csv
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any


def softmax(x):
    """Compute softmax values for array of scores."""
    exp_scores = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_scores / exp_scores.sum()


def load_predictions(filepath: str) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def get_label_name(label_idx: int) -> str:
    """Convert label index to human-readable name."""
    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    return label_map.get(label_idx, f'unknown_{label_idx}')


def extract_errors(predictions: List[Dict], source_name: str) -> List[Dict]:
    """Extract error examples from predictions."""
    errors = []
    
    for pred in predictions:
        true_label = pred['label']
        predicted_label = pred['predicted_label']
        
        # Skip correct predictions
        if true_label == predicted_label:
            continue
            
        # Calculate confidence from softmax of scores
        scores = np.array(pred['predicted_scores'])
        probs = softmax(scores)
        confidence = probs[predicted_label]
        
        # Extract relevant fields
        error_entry = {
            'source': source_name,
            'premise': pred['premise'],
            'hypothesis': pred['hypothesis'],
            'true_label': get_label_name(true_label),
            'predicted_label': get_label_name(predicted_label),
            'confidence': f"{confidence:.4f}",
            'entailment_prob': f"{probs[0]:.4f}",
            'neutral_prob': f"{probs[1]:.4f}",
            'contradiction_prob': f"{probs[2]:.4f}"
        }
        
        # Add round information for ANLI if available
        if 'round' in pred:
            error_entry['round'] = pred['round']
            
        # Add instruction for contrast sets if available
        if 'instruction' in pred:
            # Shorten instruction for readability
            if 'logically inferred' in pred['instruction']:
                error_entry['instruction_type'] = 'entailment-seeking'
            else:
                error_entry['instruction_type'] = 'non-entailment-seeking'
                
        errors.append(error_entry)
    
    return errors


def main():
    parser = argparse.ArgumentParser(description='Analyze NLI model prediction errors')
    parser.add_argument('--contrast', type=str, 
                        default='eval_output_contrast/snli_contrast_predictions.jsonl',
                        help='Path to contrast set predictions')
    parser.add_argument('--anli', type=str,
                        default='eval_output_anli/anli_predictions.jsonl',
                        help='Path to ANLI predictions')
    parser.add_argument('--output', type=str, default='error_analysis.tsv',
                        help='Output TSV file path')
    parser.add_argument('--max-errors-per-source', type=int, default=None,
                        help='Maximum number of errors to include per source')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence threshold for errors to include')
    parser.add_argument('--sort-by', choices=['confidence', 'source', 'none'], 
                        default='confidence',
                        help='How to sort the errors')
    
    args = parser.parse_args()
    
    all_errors = []
    
    # Load contrast set errors if file exists
    if Path(args.contrast).exists():
        print(f"Loading contrast set predictions from {args.contrast}")
        contrast_preds = load_predictions(args.contrast)
        contrast_errors = extract_errors(contrast_preds, 'SNLI-Contrast')
        print(f"Found {len(contrast_errors)} errors in contrast set")
        
        if args.max_errors_per_source:
            contrast_errors = contrast_errors[:args.max_errors_per_source]
        all_errors.extend(contrast_errors)
    else:
        print(f"Contrast file not found: {args.contrast}")
    
    # Load ANLI errors if file exists
    if Path(args.anli).exists():
        print(f"Loading ANLI predictions from {args.anli}")
        anli_preds = load_predictions(args.anli)
        anli_errors = extract_errors(anli_preds, 'ANLI')
        print(f"Found {len(anli_errors)} errors in ANLI")
        
        if args.max_errors_per_source:
            anli_errors = anli_errors[:args.max_errors_per_source]
        all_errors.extend(anli_errors)
    else:
        print(f"ANLI file not found: {args.anli}")
    
    # Filter by confidence if specified
    if args.min_confidence > 0:
        all_errors = [e for e in all_errors if float(e['confidence']) >= args.min_confidence]
        print(f"Filtered to {len(all_errors)} errors with confidence >= {args.min_confidence}")
    
    # Sort errors
    if args.sort_by == 'confidence':
        all_errors.sort(key=lambda x: float(x['confidence']), reverse=True)
    elif args.sort_by == 'source':
        all_errors.sort(key=lambda x: x['source'])
    
    # Write to TSV
    if all_errors:
        print(f"\nWriting {len(all_errors)} errors to {args.output}")
        
        # Determine fieldnames based on available fields
        fieldnames = ['source', 'premise', 'hypothesis', 'true_label', 'predicted_label', 
                     'confidence', 'entailment_prob', 'neutral_prob', 'contradiction_prob']
        
        # Add optional fields if they exist
        if any('round' in e for e in all_errors):
            fieldnames.append('round')
        if any('instruction_type' in e for e in all_errors):
            fieldnames.append('instruction_type')
        
        with open(args.output, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t', 
                                   extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_errors)
        
        # Print summary statistics
        print("\nError Summary:")
        print("-" * 50)
        
        # Group by source
        sources = {}
        for error in all_errors:
            src = error['source']
            if 'round' in error:
                src = f"{src}-{error['round']}"
            sources[src] = sources.get(src, 0) + 1
        
        for source, count in sorted(sources.items()):
            print(f"{source}: {count} errors")
        
        # Confusion matrix for errors
        print("\nError confusion patterns (true -> predicted):")
        confusion = {}
        for error in all_errors:
            key = f"{error['true_label']} -> {error['predicted_label']}"
            confusion[key] = confusion.get(key, 0) + 1
        
        for pattern, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
        
        # Average confidence for errors
        avg_confidence = np.mean([float(e['confidence']) for e in all_errors])
        print(f"\nAverage confidence for errors: {avg_confidence:.4f}")
        
        # High confidence errors (>0.8)
        high_conf_errors = [e for e in all_errors if float(e['confidence']) > 0.8]
        print(f"High-confidence errors (>0.8): {len(high_conf_errors)} ({len(high_conf_errors)/len(all_errors)*100:.1f}%)")
        
    else:
        print("No errors found to analyze!")


if __name__ == "__main__":
    main()