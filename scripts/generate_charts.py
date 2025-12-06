#!/usr/bin/env python3
"""
Generate charts for the research paper based on NLI model evaluation results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import csv
from pathlib import Path

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def create_snli_calibration_metrics():
    """Create SNLI calibration metrics chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Reliability Diagram (top-left)
    # Based on confidence distribution table
    conf_centers = [0.995, 0.97, 0.925, 0.85, 0.6]
    accuracies_control = [0.974, 0.869, 0.767, 0.645, 0.507]
    accuracies_contrastive = [0.985, 0.953, 0.847, 0.733, 0.641]
    counts_control = [5830, 2610, 610, 380, 580]  # Proportional to percentages
    counts_contrastive = [3250, 3700, 1010, 880, 1160]
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    # Control model
    sizes_control = [c/20 for c in counts_control]
    ax1.scatter(conf_centers, accuracies_control, s=sizes_control, alpha=0.6, 
                color='coral', label='Control', edgecolors='black', linewidth=0.5)
    # Contrastive model
    sizes_contrastive = [c/20 for c in counts_contrastive]
    ax1.scatter(conf_centers, accuracies_contrastive, s=sizes_contrastive, alpha=0.6,
                color='skyblue', label='Contrastive', edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ECE Improvement (top-right)
    # Show the improvement in calibration metrics
    metrics = ['ECE', 'MCE', 'NLL']
    improvements = [45.7, 41.6, 13.8]  # Percentage improvements from paper
    colors = ['#2E7D32', '#1B5E20', '#004D40']
    
    bars = ax2.bar(metrics, improvements, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Calibration Metric Improvements')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylim(0, 50)
    
    # Confidence Distribution (bottom-left)
    # From the paper's confidence distribution table
    ranges = ['≤0.80', '0.80-0.90', '0.90-0.95', '0.95-0.99', '>0.99']
    control_pcts = [5.8, 3.8, 6.1, 26.1, 58.3]
    contrastive_pcts = [11.6, 8.8, 10.1, 37.0, 32.5]
    
    x = np.arange(len(ranges))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, control_pcts, width, label='Control', 
                    color='coral', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, contrastive_pcts, width, label='Contrastive', 
                    color='skyblue', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('Confidence Range')
    ax3.set_ylabel('Percentage of Predictions (%)')
    ax3.set_title('Confidence Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ranges, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Calibration Metrics Comparison (bottom-right)
    metrics = ['ECE', 'MCE', 'NLL', 'Brier']
    control_vals = [0.066, 0.214, 0.370, 0.176]
    contrastive_vals = [0.036, 0.125, 0.319, 0.167]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, control_vals, width, label='Control', 
            color='coral', alpha=0.8, edgecolor='black')
    ax4.bar(x + width/2, contrastive_vals, width, label='Contrastive', 
            color='skyblue', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Value')
    ax4.set_title('Calibration Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('snli_calibration_metrics.pdf', bbox_inches='tight')
    plt.savefig('snli_calibration_metrics.png', bbox_inches='tight')
    print("Generated: snli_calibration_metrics.pdf/png")

def create_confidence_distribution():
    """Create confidence distribution comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate distributions based on actual percentages from paper
    np.random.seed(42)
    total_samples = 10000
    
    # Control distribution (58.3% >0.99, 26.1% 0.95-0.99, etc.)
    control_conf = np.concatenate([
        np.random.uniform(0.99, 1.0, int(0.583 * total_samples)),  # 58.3% >0.99
        np.random.uniform(0.95, 0.99, int(0.261 * total_samples)), # 26.1% 0.95-0.99
        np.random.uniform(0.90, 0.95, int(0.061 * total_samples)), # 6.1% 0.90-0.95
        np.random.uniform(0.80, 0.90, int(0.038 * total_samples)), # 3.8% 0.80-0.90
        np.random.uniform(0.33, 0.80, int(0.058 * total_samples))  # 5.8% <0.80
    ])
    
    # Contrastive distribution (32.5% >0.99, 37.0% 0.95-0.99, etc.)
    contrastive_conf = np.concatenate([
        np.random.uniform(0.99, 1.0, int(0.325 * total_samples)),  # 32.5% >0.99
        np.random.uniform(0.95, 0.99, int(0.370 * total_samples)), # 37.0% 0.95-0.99
        np.random.uniform(0.90, 0.95, int(0.101 * total_samples)), # 10.1% 0.90-0.95
        np.random.uniform(0.80, 0.90, int(0.088 * total_samples)), # 8.8% 0.80-0.90
        np.random.uniform(0.33, 0.80, int(0.116 * total_samples))  # 11.6% <0.80
    ])
    
    # Control distribution histogram
    ax1.hist(control_conf, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax1.axvline(x=np.mean(control_conf), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(control_conf):.3f}')
    ax1.axvline(x=0.99, color='darkred', linestyle=':', alpha=0.5, 
                label='58.3% >0.99')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Control Model (+1 epoch, α=0)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.3, 1.0)
    
    # Contrastive distribution histogram
    ax2.hist(contrastive_conf, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=np.mean(contrastive_conf), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(contrastive_conf):.3f}')
    ax2.axvline(x=0.99, color='darkblue', linestyle=':', alpha=0.5,
                label='32.5% >0.99')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Contrastive Model (+1 epoch, α=0.5)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.3, 1.0)
    
    plt.tight_layout()
    plt.savefig('confidence_distribution.pdf', bbox_inches='tight')
    plt.savefig('confidence_distribution.png', bbox_inches='tight')
    print("Generated: confidence_distribution.pdf/png")

def create_anli_results():
    """Create ANLI results comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = ['ANLI-R1', 'ANLI-R2', 'ANLI-R3']
    control_acc = [32.4, 32.0, 32.0]  # From the paper
    contrastive_acc = [32.4, 32.7, 33.0]  # From the paper
    improvements = [0.0, 0.7, 1.0]  # Percentage point improvements
    
    x = np.arange(len(rounds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_acc, width, label='Control', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, contrastive_acc, width, label='Contrastive', color='skyblue', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add improvement annotations
    for i, (r, imp) in enumerate(zip(rounds, improvements)):
        if imp > 0:
            y_pos = max(control_acc[i], contrastive_acc[i]) + 0.5
            ax.annotate(f'+{imp:.1f}%', xy=(i, y_pos), ha='center', 
                       fontsize=9, color='green', fontweight='bold')
    
    ax.set_xlabel('ANLI Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance on Adversarial NLI Dataset (Out-of-Distribution)')
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.set_ylim(30, 35)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('anli_results.pdf', bbox_inches='tight')
    plt.savefig('anli_results.png', bbox_inches='tight')
    print("Generated: anli_results.pdf/png")

def create_error_analysis_charts():
    """Create error analysis charts."""
    # Primary Lexical Errors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # From the paper's error analysis tables
    error_types = ['Negation\nFlips', 'Quantifier\nChanges', 'Antonym\nSubstitutions']
    control_errors = [18, 164, 57]  # Actual values from paper
    contrastive_errors = [19, 174, 59]  # Actual values from paper
    control_pcts = [1.7, 15.9, 5.5]  # Percentages from paper
    contrastive_pcts = [1.9, 17.0, 5.8]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_errors, width, label='Control', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, contrastive_errors, width, label='Contrastive', color='skyblue', alpha=0.8)
    
    # Add percentage labels
    for i, (bars, pcts) in enumerate([(bars1, control_pcts), (bars2, contrastive_pcts)]):
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Number of Errors')
    ax.set_title('Primary Targets (Lexical Contrasts) - Error Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(error_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('primary_lexical_errors.pdf', bbox_inches='tight')
    plt.savefig('primary_lexical_errors.png', bbox_inches='tight')
    print("Generated: primary_lexical_errors.pdf/png")
    
    # Secondary Semantic Errors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # From the paper's secondary targets table
    error_types = ['Hypernym/\nHyponym', 'Modifier\nAdditions']
    control_errors = [428, 341]  # From paper
    contrastive_errors = [423, 339]  # From paper
    control_pcts = [41.6, 33.1]  # Percentages
    contrastive_pcts = [41.2, 33.0]
    
    x = np.arange(len(error_types))
    
    bars1 = ax.bar(x - width/2, control_errors, width, label='Control', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, contrastive_errors, width, label='Contrastive', color='skyblue', alpha=0.8)
    
    # Add percentage labels
    for i, (bars, pcts) in enumerate([(bars1, control_pcts), (bars2, contrastive_pcts)]):
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Error Type')
    ax.set_ylabel('Number of Errors')
    ax.set_title('Secondary Targets (Semantic Granularity) - Error Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(error_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('secondary_semantic_errors.pdf', bbox_inches='tight')
    plt.savefig('secondary_semantic_errors.png', bbox_inches='tight')
    print("Generated: secondary_semantic_errors.pdf/png")

def create_premise_group_errors():
    """Create premise grouping error analysis chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # From the paper's contrastive premise groups table
    categories = ['High-contrast\nPremises', 'Low-contrast\nPremises']
    control_errors = [141, 233]  # From paper
    contrastive_errors = [145, 230]  # From paper
    control_pcts = [13.7, 22.6]
    contrastive_pcts = [14.1, 22.4]
    changes = [2.8, -1.3]  # Percentage changes
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_errors, width, label='Control', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, contrastive_errors, width, label='Contrastive', color='skyblue', alpha=0.8)
    
    # Add percentage labels
    for i, (bars, pcts) in enumerate([(bars1, control_pcts), (bars2, contrastive_pcts)]):
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Add change annotations
    for i, change in enumerate(changes):
        y_pos = max(control_errors[i], contrastive_errors[i]) + 5
        color = 'green' if change < 0 else 'red'
        ax.annotate(f'{change:+.1f}%', xy=(i, y_pos), ha='center',
                   fontsize=9, color=color, fontweight='bold')
    
    ax.set_xlabel('Premise Group Type')
    ax.set_ylabel('Number of Errors')
    ax.set_title('Contrastive Premise Groups - Error Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('premise_group_errors.pdf', bbox_inches='tight')
    plt.savefig('premise_group_errors.png', bbox_inches='tight')
    print("Generated: premise_group_errors.pdf/png")

def create_overlap_analysis():
    """Create lexical overlap analysis chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # From the paper's overlap-based analysis table
    overlap_categories = ['Very High\n(>60%)', 'Very Low\n(<20%)']
    control_errors = [35, 547]
    contrastive_errors = [32, 554]
    control_pcts = [3.4, 53.1]
    contrastive_pcts = [3.1, 54.0]
    changes = [-8.6, 1.3]
    
    x = np.arange(len(overlap_categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, control_errors, width, label='Control', color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, contrastive_errors, width, label='Contrastive', color='skyblue', alpha=0.8)
    
    # Add percentage labels
    for i, (bars, pcts) in enumerate([(bars1, control_pcts), (bars2, contrastive_pcts)]):
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Add change annotations
    for i, change in enumerate(changes):
        y_pos = max(control_errors[i], contrastive_errors[i]) + 10
        color = 'green' if change < 0 else 'red'
        ax1.annotate(f'{change:+.1f}%', xy=(i, y_pos), ha='center',
                    fontsize=9, color=color, fontweight='bold')
    
    ax1.set_xlabel('Overlap Level')
    ax1.set_ylabel('Number of Errors')
    ax1.set_title('Error Distribution by Lexical Overlap')
    ax1.set_xticks(x)
    ax1.set_xticklabels(overlap_categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # False entailment/contradiction analysis
    error_types = ['False Entailment\n(high overlap)', 'False Contradiction\n(low overlap)']
    control_errors2 = [14, 153]
    contrastive_errors2 = [13, 164]
    control_pcts2 = [1.4, 14.9]
    contrastive_pcts2 = [1.3, 16.0]
    changes2 = [-7.1, 7.2]
    
    x2 = np.arange(len(error_types))
    
    bars3 = ax2.bar(x2 - width/2, control_errors2, width, label='Control', color='coral', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, contrastive_errors2, width, label='Contrastive', color='skyblue', alpha=0.8)
    
    # Add percentage labels
    for i, (bars, pcts) in enumerate([(bars3, control_pcts2), (bars4, contrastive_pcts2)]):
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Add change annotations
    for i, change in enumerate(changes2):
        y_pos = max(control_errors2[i], contrastive_errors2[i]) + 3
        color = 'green' if change < 0 else 'red'
        ax2.annotate(f'{change:+.1f}%', xy=(i, y_pos), ha='center',
                    fontsize=9, color=color, fontweight='bold')
    
    ax2.set_xlabel('Error Type')
    ax2.set_ylabel('Number of Errors')
    ax2.set_title('False Predictions by Overlap Pattern')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(error_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('overlap_analysis.pdf', bbox_inches='tight')
    plt.savefig('overlap_analysis.png', bbox_inches='tight')
    print("Generated: overlap_analysis.pdf/png")

def main():
    """Generate all charts for the research paper."""
    print("Generating research paper charts...")
    
    # Create all charts
    create_snli_calibration_metrics()
    create_confidence_distribution()
    create_anli_results()
    create_error_analysis_charts()
    create_premise_group_errors()
    create_overlap_analysis()
    
    print("\nAll charts generated successfully!")
    print("Files created:")
    print("  - snli_calibration_metrics.pdf/png")
    print("  - confidence_distribution.pdf/png")
    print("  - anli_results.pdf/png")
    print("  - primary_lexical_errors.pdf/png")
    print("  - secondary_semantic_errors.pdf/png")
    print("  - premise_group_errors.pdf/png")
    print("  - overlap_analysis.pdf/png")

if __name__ == "__main__":
    main()