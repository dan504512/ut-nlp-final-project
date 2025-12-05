# Adversarial Fine-Tuning of ELECTRA for NLI Robustness

## Abstract
We investigate how adversarial fine-tuning with ANLI affects the robustness of an ELECTRA-small NLI model originally trained on SNLI. Starting from a strong SNLI baseline (≈89% accuracy), we add an ANLI fine-tuning stage and evaluate on both SNLI and ANLI (rounds A1–A3). ANLI fine-tuning improves combined ANLI accuracy by +10.8 points over a control model (SNLI→SNLI) while keeping SNLI accuracy unchanged, indicating better robustness without sacrificing in-domain performance.

## 1 Introduction
Natural Language Inference models can achieve high accuracy on standard benchmarks such as SNLI. But these benchmarks often fail to accurately measure a model’s true language understanding capabilities and rather rely on “spurious correlations” or dataset artifacts (lexical overlap, negation words). Adversarial challenge sets like ANLI deliberately break the shortcut strategies models learn. We ask: (1) How poorly does a strong SNLI model behave on ANLI? (2) Does a brief ANLI fine-tuning stage improve out-of-domain robustness without hurting SNLI?

**Contributions**
- Establish a SNLI ELECTRA-small baseline and quantify its collapse on ANLI.
- Compare a control run (extra SNLI epoch) to an adversarial run (SNLI→SNLI+ANLI).
- Show ANLI fine-tuning raises ANLI accuracy by 34% relative (+10.8 absolute) with no SNLI loss, and give qualitative error fixes.

## 2 Background and Related Work
Background on ELECTRA: ELECTRA-small is an encoder-only discriminative model trained to detect replaced tokens, making it suitable for classification-based NLP tasks such as NLI. Adversarial challenge datasets are created using a human-and-model-in-the-loop procedure where human annotators construct examples designed to expose the model’s weaknesses via subtle perturbations:
- Lexical perturbations (swapping related words or antonyms)
- Composition structure changes
- World knowledge contradictions
- Negation insertions

Adversarial example (ANLI):  
Premise: “World Premiere is a 1941 American comedy film … released on August 21, 1941, by Paramount Pictures.”  
Hypothesis: “World Premiere was not released by Universal Studios.”  
Gold: entailment; Baseline prediction: contradiction.  
The model latches onto “not” and misses that “released by Paramount” is compatible with “not released by Universal.”

**Table 0: ANLI dataset splits (Nie et al., 2020)**

| Round | Context source                                   | Train | Dev  | Test |
|-------|--------------------------------------------------|------:|-----:|-----:|
| A1    | Wikipedia passages                               | 16,946| 1,000| 1,000|
| A2    | New non-overlapping Wikipedia passages           | 45,460| 1,000| 1,000|
| A3    | Wikipedia + News + Fiction + Spoken + WikiHow    |120,379| 1,400| 1,400|

We use ANLI as the adversarial signal to “fix” the base ELECTRA model.

## 3 Method / Setup
- **Model/head**: ELECTRA-small with a 3-way sequence classification head (entailment, neutral, contradiction).
- **Tokenization**: HuggingFace `AutoTokenizer`; truncate/pad premise–hypothesis pairs to max length 128.
- **Hyperparameters**: `per_device_train_batch_size=8`, `per_device_eval_batch_size=8`, learning rate 5e-5, linear scheduler, no warmup, AdamW (β1=0.9, β2=0.999, ε=1e-8), weight decay 0.0, max grad norm 1.0. Initial SNLI training uses 3 epochs; continued fine-tuning uses 1 epoch.
- **Data**: SNLI (filtered to drop unlabeled items). ANLI concatenates train_r1/2/3 for training and dev_r1/2/3 for validation; cast to common schema (premise, hypothesis, label) and drop ANLI-only columns (e.g., uid, reason). For joint SNLI+ANLI, we concatenate SNLI train with ANLI train and SNLI dev with ANLI dev.
- **Metric**: Accuracy computed via HuggingFace Trainer on validation/test splits; ANLI reported per round and combined.

## 4 Experimental Design
Our approach involved a three-stage fine-tuning process:
1) **Baseline (SNLI-only)**: ELECTRA-small trained 3 epochs on SNLI; evaluate on SNLI and ANLI test rounds (A1–A3) for robustness.
2) **Control (SNLI→SNLI)**: Starting from the baseline SNLI-finetuned checkpoint, fine-tune 1 additional epoch on SNLI only; evaluate as above to isolate “just more SNLI training.”
3) **Fixed (SNLI→SNLI+ANLI)**: Starting from the baseline checkpoint, fine-tune 1 epoch on concatenated SNLI+ANLI; validate on concatenated SNLI dev + ANLI dev; evaluate on SNLI and ANLI test rounds.

All runs use max_length=128 and default hyperparameters unless stated. Qualitative analysis inspects error types (negation, lexical overlap, world knowledge).

## 5 Results
**Table 1: Test accuracy (%)**

| Model (regime)            | SNLI | ANLI A1 | ANLI A2 | ANLI A3 | ANLI combined |
|---------------------------|-----:|--------:|--------:|--------:|--------------:|
| Baseline (SNLI-only)      | 89.29 |   30.70 |   30.40 |   30.83 |         30.64 |
| Control (SNLI→SNLI)       | 89.53 |   31.40 |   31.60 |   31.83 |         31.61 |
| Fixed (SNLI→SNLI+ANLI)    | 89.39 |   47.40 |   39.50 |   40.25 |         42.38 |

- In-domain (SNLI) performance (Table 1): All three models perform nearly identically on SNLI. The control model increases SNLI accuracy by only +0.24 points, indicating the baseline is near its ceiling. The fixed model, despite being trained on harder ANLI data, maintains virtually the same SNLI accuracy (–0.14 vs control). Adversarial fine-tuning does not harm in-domain performance; differences are not significant.
- Out-of-domain (ANLI) performance (Table 1): Baseline collapses on ANLI (≈30.6%, essentially random for 3-way). Control is only +0.97 better. Fixed (SNLI+ANLI) improves dramatically: +16 on A1, +7.9 on A2, +8.42 on A3, +10.77 combined (+34% relative). Adversarial fine-tuning massively increases robustness on all rounds while preserving SNLI performance. Figure 1 visualizes the SNLI/ANLI trade-off, and Figure 2 breaks down ANLI improvements by round.

![Figure 1: SNLI vs ANLI accuracy across models](figs/image1.png)

![Figure 2: ANLI round-level accuracy comparison](figs/image2.png)

## 6 Error Analysis (qualitative)
Here are some errors present in Control evaluations but not in the Fixed evaluations (Table 2):

**Table 2: Qualitative errors fixed by ANLI fine-tuning**

| Premise (abridged)                                                     | Hypothesis                                           | Gold | Control pred | Error type        |
|------------------------------------------------------------------------|------------------------------------------------------|------|--------------|-------------------|
| World Premiere … released by Paramount Pictures                        | World Premiere was not released by Universal Studios | E    | C            | Negation trap     |
| The Second Jungle Book … not based on “The Second Jungle Book”         | The first Jungle Book was written before 1997        | E    | C            | World knowledge   |
| Beat TV … daily entertainment show with various celebrity guests       | The hosts on Beat TV shared the screen time equally  | N    | E            | Lexical overlap   |

E = entailment, N = neutral, C = contradiction.

## 7 Discussion and Limitations
- ANLI fine-tuning materially improves robustness while preserving SNLI accuracy, showing adversarial data helps without overfitting away in-domain performance.
- Gains are smaller on A2/A3 than A1, likely due to harder domains (news, fiction, spoken transcripts) and limited model capacity (ELECTRA-small).
- Threats to validity: single model size, single fine-tuning epoch for ANLI, no variance reported; results may be sensitive to learning rate/epochs.

## 8 Conclusion and Future Work
Brief adversarial fine-tuning (SNLI→SNLI+ANLI) boosts ANLI robustness by ~11 points with no SNLI cost. Future work: explore larger models, longer ANLI schedules, curriculum mixing of SNLI/ANLI, calibration or debiasing methods, and richer behavioral tests (contrast/checklist).

## References
- Clark et al. 2020. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.
- Bowman et al. 2015. A large annotated corpus for learning natural language inference (SNLI).
- Nie et al. 2020. Adversarial NLI: A new benchmark for natural language understanding.
- Poliak et al. 2018; McCoy et al. 2019. Analyses of NLI artifacts and syntactic heuristics.
