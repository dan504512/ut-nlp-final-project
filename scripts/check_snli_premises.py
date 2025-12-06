#!/usr/bin/env python
"""Check if SNLI has multiple hypotheses for the same premise."""

import datasets
from collections import defaultdict

# Load SNLI dataset
print("Loading SNLI dataset...")
snli = datasets.load_dataset('snli')

# Analyze train split
train = snli['train'].filter(lambda ex: ex['label'] != -1)  # Remove no-label examples

print(f"\nAnalyzing SNLI train split...")
print(f"Total examples: {len(train)}")

# Group by premise
premise_groups = defaultdict(list)
for i, ex in enumerate(train):
    if i < 50000:  # Sample first 50k for speed
        premise_groups[ex['premise']].append({
            'hypothesis': ex['hypothesis'],
            'label': ex['label']
        })

# Analyze premise groups
group_sizes = defaultdict(int)
for premise, hypotheses in premise_groups.items():
    group_sizes[len(hypotheses)] += 1

print(f"\nUnique premises (in first 50k): {len(premise_groups)}")
print("\nDistribution of hypotheses per premise:")
for size in sorted(group_sizes.keys()):
    count = group_sizes[size]
    print(f"  {size} hypotheses: {count} premises ({count/len(premise_groups)*100:.1f}%)")

# Show examples of premises with multiple hypotheses
print("\n" + "="*80)
print("EXAMPLES OF PREMISES WITH MULTIPLE HYPOTHESES")
print("="*80)

LABEL_MAP = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

# Find premises with exactly 3 hypotheses (ideal for contrastive learning)
good_examples = [p for p, h in premise_groups.items() if len(h) == 3]

for i, premise in enumerate(good_examples[:5]):
    print(f"\n--- Example {i+1} ---")
    print(f"PREMISE: {premise}")
    print("\nHYPOTHESES:")
    for j, hyp_data in enumerate(premise_groups[premise]):
        label_name = LABEL_MAP[hyp_data['label']]
        print(f"  {j+1}. {hyp_data['hypothesis']}")
        print(f"     Label: {label_name}")

# Find premises with different label distributions
print("\n" + "="*80)
print("PREMISES WITH DIVERSE LABELS (one of each)")
print("="*80)

for premise, hypotheses in premise_groups.items():
    labels = [h['label'] for h in hypotheses]
    if len(hypotheses) == 3 and set(labels) == {0, 1, 2}:
        print(f"\nPREMISE: {premise}")
        for j, hyp_data in enumerate(hypotheses):
            label_name = LABEL_MAP[hyp_data['label']]
            print(f"  {j+1}. {hyp_data['hypothesis']}")
            print(f"     Label: {label_name}")
        break

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print("""
SNLI naturally has multiple hypotheses for many premises!
- Most premises have 2-3 different hypotheses
- These hypotheses often have different labels (entailment, neutral, contradiction)
- This makes SNLI perfect for contrastive learning WITHOUT needing a separate contrast dataset

We can create bundles by:
1. Grouping all examples by premise
2. Using each group as a natural bundle for contrastive learning
3. The model learns to distinguish between different inferences from the same premise
""")