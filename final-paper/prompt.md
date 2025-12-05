Here is my research paper,
followed by my partners research paper.
I need to combine them. Not the main bodies,
just abstract/introduction/background/related work/conclusion/future work/references.
Generate those combined sections for me, excluding the other sections.

# Improving Calibration in Natural Language Inference through 
# Premise-Bundled Contrastive Learning

## Abstract

Natural language inference (NLI) models often exhibit overconfidence, 
particularly on incorrect predictions. We present a contrastive learning 
approach that leverages naturally occurring premise-hypothesis bundles in SNLI 
to improve model calibration. Our method introduces a margin ranking loss with 
lexical-overlap-based dynamic weighting, encouraging models to produce less 
peaked distributions when facing challenging distinctions. While overall 
accuracy remains virtually unchanged on SNLI (89.53% to 89.58%), our approach 
dramatically improves calibration: Expected Calibration Error (ECE) decreases 
by 45.7% (0.066 to 0.036), Maximum Calibration Error drops by 41.6%, and the 
proportion of overconfident predictions (>0.99 confidence) falls from 58.3% to 
32.5%. Notably, extreme confidence errors drop from 14.5% to 4.7% of all 
errors. On the adversarial ANLI benchmark, we observe small but consistent 
improvements on harder rounds (R2: +0.7%, R3: +1.0%). Our findings demonstrate 
that contrastive fine-tuning effectively redistributes model confidence while 
maintaining accuracy, offering a computationally efficient calibration-aware 
fine-tuning approach.

## 1. Introduction

Pre-trained language models fine-tuned on NLI benchmarks like SNLI (Bowman 
et al., 2015) often achieve impressive accuracy scores. However, these models 
frequently suffer from poor calibration, producing overconfident predictions 
even when they are incorrect (Guo et al., 2017). This overconfidence can be 
particularly problematic in deployment scenarios where prediction 
probabilities are used for decision-making or uncertainty estimation.

While dataset artifactsâspurious correlations between surface features and 
labelsâhave been well-documented in NLI (Poliak et al., 2018; McCoy et al., 
2019), improving model calibration represents an orthogonal but equally 
important challenge. Well-calibrated models that accurately reflect their 
uncertainty can be more trustworthy in practice, even if their accuracy 
remains unchanged.

Recent work has explored various approaches to improve NLI robustness, 
including ensemble methods (Clark et al., 2019), adversarial training (Liu et 
al., 2019), and dataset cartography (Swayamdipta et al., 2020). Contrastive 
learning has shown promise in reading comprehension (Dua et al., 2021) but 
its impact on calibration in NLI remains underexplored.

This work investigates whether contrastive learning can improve calibration 
in NLI models by exploiting the natural structure of SNLI, where multiple 
hypotheses are paired with the same premise. We introduce a dynamic weighting 
scheme based on lexical overlap between hypotheses, hypothesizing that 
forcing models to produce appropriate uncertainty when distinguishing between 
similar hypotheses will improve calibration. Our contributions are:

1. A contrastive learning framework that leverages premise-hypothesis bundles 
   with overlap-based dynamic weighting
2. Empirical evidence that contrastive fine-tuning dramatically improves 
   calibration (ECE: -45.7%, MCE: -41.6%, NLL: -13.8%) with minimal 
   computational overhead
3. Analysis showing small but consistent improvements on adversarial ANLI, 
   suggesting potential robustness benefits
4. Comprehensive error analysis revealing mixed effects on specific linguistic 
   phenomena, indicating complex redistribution of model errors

## 2. Related Work

### 2.1 Dataset Artifacts in NLI

Multiple studies have demonstrated that NLI models exploit annotation 
artifacts. Poliak et al. (2018) showed that hypothesis-only baselines achieve 
surprisingly high accuracy, suggesting models rely on hypothesis-specific 
biases. Gardner et al. (2020) introduced contrast setsâminimally edited 
examples that flip the labelârevealing brittleness in model predictions. 
McCoy et al. (2019) identified specific syntactic heuristics that models 
incorrectly learn, such as assuming lexical overlap implies entailment.

### 2.2 Debiasing Approaches

Various strategies have been proposed to address dataset artifacts:

**Ensemble Methods**: Clark et al. (2019) train a weak model on biased 
features, then train the main model on the residual. He et al. (2019) use 
similar residual fitting approaches.

**Data Augmentation**: Liu et al. (2019) propose "inoculation by 
fine-tuning" using challenge examples. Morris et al. (2020) generate 
adversarial examples for training.

**Training Modifications**: Swayamdipta et al. (2020) use dataset cartography 
to identify and upweight hard examples. Utama et al. (2020) propose 
confidence regularization to prevent overreliance on biases.

### 2.3 Contrastive Learning

Contrastive learning has been successful in representation learning (Chen et 
al., 2020) and has recently been applied to NLP tasks. Dua et al. (2021) use 
instance bundles for reading comprehension, showing improvements on 
adversarial evaluation sets. Our work extends this approach to NLI with a 
novel dynamic weighting scheme.

## 3. Methodology

### 3.1 Problem Formulation

Given a premise $p$ and hypothesis $h$, an NLI model predicts a label 
$y \in \{entailment, neutral, contradiction\}$. Standard training minimizes 
cross-entropy loss:

$$\mathcal{L}_{CE} = -\log P(y|p, h)$$

However, this objective doesn't explicitly encourage models to distinguish 
between similar hypotheses with different labels, allowing them to rely on 
superficial patterns.

### 3.2 Contrastive Learning Framework

We leverage SNLI's annotation structure, where multiple hypotheses are 
written for each premise. For a premise $p$ with hypotheses 
$\{h_1, ..., h_n\}$ and labels $\{y_1, ..., y_n\}$, we create contrastive 
bundles. For each pair $(h_i, h_j)$ where $y_i \neq y_j$, we add a margin 
ranking loss:

$$\mathcal{L}_{margin}^{i,j} = \max(0, m - (s_i^{y_i} - s_j^{y_i})) + 
\max(0, m - (s_j^{y_j} - s_i^{y_j}))$$

where $s_i^{y_i}$ is hypothesis $i$'s score for its true label $y_i$, and 
$m$ is the margin.

### 3.3 Dynamic Weighting Based on Lexical Overlap

We hypothesize that hypothesis pairs with high lexical overlap but different 
labels are particularly informative for learning robust distinctions. We 
compute Jaccard similarity between hypotheses after removing stopwords and 
stemming:

$$overlap(h_i, h_j) = \frac{|words(h_i) \cap words(h_j)|}{|words(h_i) \cup 
words(h_j)|}$$

The dynamic weight for each pair is:

$$w_{i,j} = 0.5 + 1.5 \times overlap(h_i, h_j)$$

This maps overlap scores (0-1) to weights (0.5-2.0), emphasizing high-overlap 
contrasts.

### 3.4 Combined Objective

The final training objective combines cross-entropy and weighted margin 
losses:

$$\mathcal{L} = (1-\alpha)\mathcal{L}_{CE} + \alpha \sum_{i,j: y_i \neq y_j} 
w_{i,j} \mathcal{L}_{margin}^{i,j}$$

where $\alpha$ controls the contribution of contrastive learning.

## 4. Experimental Setup

### 4.1 Dataset and Model

We use the Stanford Natural Language Inference (SNLI) dataset (Bowman et 
al., 2015) containing 550K training, 10K validation, and 10K test examples. 
We fine-tune ELECTRA-small (Clark et al., 2020), a 14M parameter model that 
achieves strong performance with computational efficiency.

### 4.2 Training Procedure

We follow a two-stage training approach:

1. **Baseline Training**: Fine-tune ELECTRA-small on SNLI for 3 epochs using 
   standard cross-entropy loss
2. **Contrastive Fine-tuning**: Continue training for 1 additional epoch with 
   our contrastive objective

For the experimental condition, we set $\alpha = 0.5$ and margin $m = 0.5$. 
The control condition uses $\alpha = 0$ (standard fine-tuning for one 
additional epoch). We use batch size 32, learning rate 5e-5 with linear 
warmup, and AdamW optimizer.

### 4.3 Evaluation

We evaluate on the SNLI validation set and the Adversarial NLI (ANLI) test 
sets to assess both in-domain and out-of-distribution performance. We analyze 
errors using:
- Overall accuracy on SNLI and ANLI
- Confidence calibration metrics
- Fine-grained error categorization based on linguistic phenomena
- Performance on high vs. low lexical overlap subsets

## 5. Results

### 5.1 Overall Performance and Calibration

#### SNLI Validation Set

| Model | Accuracy | ECE | MCE | NLL | Brier |
|-------|----------|-----|-----|-----|-------|
| Control (+1 epoch, $\alpha$=0) | 89.53% | 0.066 | 0.214 | 0.370 | 0.176 |
| Contrastive (+1 epoch, $\alpha$=0.5) | 89.58% | 0.036 | 0.125 | 0.319 | 0.167 |

The calibration metrics reveal substantial improvements despite minimal 
accuracy gains (+0.05%):

- **Expected Calibration Error (ECE)** reduced by 45.7% (0.066 -> 0.036)
- **Maximum Calibration Error (MCE)** reduced by 41.6% (0.214 -> 0.125)
- **Negative Log-Likelihood (NLL)** improved by 13.8% (0.370 -> 0.319)
- **Brier Score** improved by 5.1% (0.176 â 0.167)

#### Confidence Distribution

| Confidence Range | Control | | Contrastive | |
|-----------------|---------|----------|------------|----------|
| | Count (%) | Accuracy | Count (%) | Accuracy |
| (0.99, 1.00] | 58.3% | 0.974 | 32.5% | 0.985 |
| (0.95, 0.99] | 26.1% | 0.869 | 37.0% | 0.953 |
| (0.90, 0.95] | 6.1% | 0.767 | 10.1% | 0.847 |
| (0.80, 0.90] | 3.8% | 0.645 | 8.8% | 0.733 |
| $\leq$ 0.80 | 5.8% | 0.507 | 11.6% | 0.641 |

The contrastive model shows a dramatic shift away from extreme confidence 
(>0.99), with predictions more evenly distributed across confidence ranges. 
The proportion of predictions with confidence >0.99 drops from 58.3% to 32.5%, 
while maintaining or improving accuracy in each confidence bin. Among errors, 
extreme confidence (>0.99) drops from 14.5% to 4.7% of all errors.

#### Adversarial NLI (Out-of-Distribution)

| Model | ANLI-R1 | ANLI-R2 | ANLI-R3 |
|-------|---------|---------|---------|
| Control | 32.4% | 32.0% | 32.0% |
| Contrastive | 32.4% | 32.7% | 33.0% |

On the challenging ANLI benchmark, contrastive learning shows small but 
consistent improvements on rounds R2 (+0.7%) and R3 (+1.0%), with no change 
on R1. These gains, though modest, suggest improved robustness on adversarial 
examples.

### 5.2 Error Analysis by Category

We categorize errors based on linguistic phenomena that contrastive learning 
should address:

#### Primary Targets (Lexical Contrasts)

| Category | Control Errors | Contrastive Errors | Change |
|----------|----------------|-------------------|---------|
| Negation flips | 18 (1.7%) | 19 (1.9%) | +5.6% |
| Quantifier changes | 164 (15.9%) | 174 (17.0%) | +6.1% |
| Antonym substitutions | 57 (5.5%) | 59 (5.8%) | +3.5% |

#### Secondary Targets (Semantic Granularity)

| Category | Control Errors | Contrastive Errors | Change |
|----------|----------------|-------------------|---------|
| Hypernym/hyponym | 428 (41.6%) | 423 (41.2%) | -1.2% |
| Modifier additions | 341 (33.1%) | 339 (33.0%) | -0.6% |

#### Contrastive Premise Groups

| Category | Control Errors | Contrastive Errors | Change |
|----------|----------------|-------------------|---------|
| High-contrast premises | 141 (13.7%) | 145 (14.1%) | +2.8% |
| Low-contrast premises | 233 (22.6%) | 230 (22.4%) | -1.3% |

### 5.3 Overlap-Based Analysis

| Overlap Level | Control Errors | Contrastive Errors | Change |
|---------------|----------------|-------------------|---------|
| Very high (>60%) | 35 (3.4%) | 32 (3.1%) | -8.6% |
| Very low (<20%) | 547 (53.1%) | 554 (54.0%) | +1.3% |
| False entailment (high overlap) | 14 (1.4%) | 13 (1.3%) | -7.1% |
| False contradiction (low overlap) | 153 (14.9%) | 164 (16.0%) | +7.2% |

## 6. Discussion

### 6.1 Calibration vs. Accuracy

The most striking finding is the comprehensive improvement in calibration 
metrics despite minimal accuracy gains. The 45.7% reduction in ECE (from 
0.066 to 0.036) brings the model much closer to perfect calibration. The 
41.6% reduction in MCE shows that even the worst-calibrated confidence bins 
are now substantially improved.

The confidence distribution analysis reveals the mechanism: the control model 
places 58.3% of predictions at extreme confidence (>0.99), while the 
contrastive model reduces this to 32.5%. This redistribution doesn't harm 
accuracyâin fact, accuracy improves in nearly every confidence bin. Most 
notably, among errors, extreme confidence (>0.99) drops from 14.5% to just 
4.7%, indicating the model has learned to be appropriately uncertain when it 
is likely to be wrong. These results suggest contrastive learning primarily 
affects confidence calibration rather than decision boundaries, making the 
model express appropriate uncertainty without sacrificing performance.

### 6.2 Mixed Results on Linguistic Phenomena

Our error analysis shows mixed results on targeted linguistic categories. 
Surprisingly, error rates for categories we expected to improve (negation, 
quantifiers, antonyms) showed slight increases. This suggests that contrastive 
learning may induce complex redistributions of errors rather than 
straightforward improvements on specific phenomena. Several factors may 
explain this:

1. **Insufficient Training**: One epoch may be too short to substantially 
   reorganize learned representations
2. **Bundle Composition**: Natural premise-hypothesis bundles may not 
   consistently contain the contrasts needed to address specific phenomena
3. **Competing Objectives**: The model must balance maintaining overall 
   accuracy while adjusting confidence distributions

### 6.3 Evidence of Selective Improvements

Despite mixed results overall, we observe selective improvements worth noting. 
The reduction in very high overlap errors (-8.6%) and false entailments from 
high overlap (-7.1%) suggests the model has learned to be more cautious about 
assuming lexical similarity implies entailment. These targeted gains align 
with our hypothesis that overlap-weighted contrastive learning would most 
benefit high-similarity distinctions.

### 6.4 Implications for Robustness

The ANLI results provide suggestive evidence that contrastive learning may 
offer mild robustness benefits, particularly on the more challenging R2 and 
R3 rounds. While these improvements are small (0.7-1.0%) and would benefit 
from significance testing across multiple seeds, they are directionally 
consistent with the hypothesis that better-calibrated models may generalize 
more reliably. However, we emphasize that these gains are incremental rather 
than transformative, and more comprehensive evaluation on targeted artifact 
benchmarks (e.g., HANS, contrast sets) would be needed to make strong claims 
about artifact mitigation.

## 7. Limitations and Future Work

### 7.1 Limitations

1. **Limited Training Duration**: Only one epoch of contrastive learning may 
   be insufficient for substantial representation changes
2. **Static Margin**: A fixed margin may not be optimal for all example pairs
3. **Bundle Quality**: Relying on naturally occurring bundles may miss 
   important contrasts
4. **Limited OOD Evaluation**: While we evaluate on ANLI, more comprehensive 
   out-of-distribution testing (e.g., HANS, contrast sets) would strengthen 
   claims about robustness
5. **Single Seed**: Results are from single runs; multiple seeds would provide 
   more reliable estimates of performance variance

### 7.2 Future Directions

1. **Direct Artifact Testing**: Evaluate on HANS and hypothesis-only baselines 
   to directly measure artifact reliance
2. **Multiple Seeds and Hyperparameters**: Test stability across random seeds 
   and explore margin/weight sensitivity
3. **Extended Training**: Investigate longer contrastive training or 
   interleaving with standard training
4. **Synthetic Contrasts**: Generate targeted contrast examples for specific 
   linguistic phenomena
5. **Fine-grained Overlap Analysis**: Stratify performance by overlap deciles 
   to understand where improvements concentrate

## 8. Conclusion

We presented a contrastive learning approach for NLI that leverages natural 
premise-hypothesis bundles with lexical-overlap-based weighting. Our primary 
finding is that contrastive fine-tuning dramatically improves model 
calibration, reducing high-confidence errors by 54.9% while maintaining 
comparable accuracy. This calibration improvement is particularly valuable for 
deployment scenarios where prediction probabilities inform downstream 
decisions.

While we observe small improvements on adversarial ANLI benchmarks, 
suggesting potential robustness benefits, our error analysis reveals mixed 
results on specific linguistic phenomena. Rather than systematically improving 
on targeted categories like negation or quantifiers, the method appears to 
induce complex redistributions of errors, with selective improvements on 
high-overlap examples.

This work demonstrates that contrastive fine-tuning offers a computationally 
efficient approach to improving calibration in NLI models. The dramatic 
calibration gains achieved with just one additional epoch of training suggest 
this approach could be valuable as a lightweight calibration-aware fine-tuning 
step after standard training. Future 
work should employ richer calibration metrics, test directly for artifact 
mitigation, and explore whether longer training or synthetic contrasts could 
yield more systematic improvements on specific linguistic phenomena.

# Adversarial Fine-Tuning of ELECTRA for NLI Robustness

## Abstract
We investigate how adversarial fine-tuning with ANLI affects the robustness of an ELECTRA-small NLI model originally trained on SNLI. Starting from a strong SNLI baseline (â89% accuracy), we add an ANLI fine-tuning stage and evaluate on both SNLI and ANLI (rounds A1âA3). ANLI fine-tuning improves combined ANLI accuracy by +10.8 points over a control model (SNLIâSNLI) while keeping SNLI accuracy unchanged, indicating better robustness without sacrificing in-domain performance.

## 1 Introduction
Natural Language Inference models can achieve high accuracy on standard benchmarks such as SNLI. But these benchmarks often fail to accurately measure a modelâs true language understanding capabilities and rather rely on âspurious correlationsâ or dataset artifacts (lexical overlap, negation words). Adversarial challenge sets like ANLI deliberately break the shortcut strategies models learn. We ask: (1) How poorly does a strong SNLI model behave on ANLI? (2) Does a brief ANLI fine-tuning stage improve out-of-domain robustness without hurting SNLI?

**Contributions**
- Establish a SNLI ELECTRA-small baseline and quantify its collapse on ANLI.
- Compare a control run (extra SNLI epoch) to an adversarial run (SNLIâSNLI+ANLI).
- Show ANLI fine-tuning raises ANLI accuracy by 34% relative (+10.8 absolute) with no SNLI loss, and give qualitative error fixes.

## 2 Background and Related Work
Background on ELECTRA: ELECTRA-small is an encoder-only discriminative model trained to detect replaced tokens, making it suitable for classification-based NLP tasks such as NLI. Adversarial challenge datasets are created using a human-and-model-in-the-loop procedure where human annotators construct examples designed to expose the modelâs weaknesses via subtle perturbations:
- Lexical perturbations (swapping related words or antonyms)
- Composition structure changes
- World knowledge contradictions
- Negation insertions

Adversarial example (ANLI):  
Premise: âWorld Premiere is a 1941 American comedy film â¦ released on August 21, 1941, by Paramount Pictures.â  
Hypothesis: âWorld Premiere was not released by Universal Studios.â  
Gold: entailment; Baseline prediction: contradiction.  
The model latches onto ânotâ and misses that âreleased by Paramountâ is compatible with ânot released by Universal.â

**Table 0: ANLI dataset splits (Nie et al., 2020)**

| Round | Context source                                   | Train | Dev  | Test |
|-------|--------------------------------------------------|------:|-----:|-----:|
| A1    | Wikipedia passages                               | 16,946| 1,000| 1,000|
| A2    | New non-overlapping Wikipedia passages           | 45,460| 1,000| 1,000|
| A3    | Wikipedia + News + Fiction + Spoken + WikiHow    |120,379| 1,400| 1,400|

We use ANLI as the adversarial signal to âfixâ the base ELECTRA model.

## 3 Method / Setup
- **Model/head**: ELECTRA-small with a 3-way sequence classification head (entailment, neutral, contradiction).
- **Tokenization**: HuggingFace `AutoTokenizer`; truncate/pad premiseâhypothesis pairs to max length 128.
- **Hyperparameters**: `per_device_train_batch_size=8`, `per_device_eval_batch_size=8`, learning rate 5e-5, linear scheduler, no warmup, AdamW (Î²1=0.9, Î²2=0.999, Îµ=1e-8), weight decay 0.0, max grad norm 1.0. Initial SNLI training uses 3 epochs; continued fine-tuning uses 1 epoch.
- **Data**: SNLI (filtered to drop unlabeled items). ANLI concatenates train_r1/2/3 for training and dev_r1/2/3 for validation; cast to common schema (premise, hypothesis, label) and drop ANLI-only columns (e.g., uid, reason). For joint SNLI+ANLI, we concatenate SNLI train with ANLI train and SNLI dev with ANLI dev.
- **Metric**: Accuracy computed via HuggingFace Trainer on validation/test splits; ANLI reported per round and combined.

## 4 Experimental Design
Our approach involved a three-stage fine-tuning process:
1) **Baseline (SNLI-only)**: ELECTRA-small trained 3 epochs on SNLI; evaluate on SNLI and ANLI test rounds (A1âA3) for robustness.
2) **Control (SNLIâSNLI)**: Starting from the baseline SNLI-finetuned checkpoint, fine-tune 1 additional epoch on SNLI only; evaluate as above to isolate âjust more SNLI training.â
3) **Fixed (SNLIâSNLI+ANLI)**: Starting from the baseline checkpoint, fine-tune 1 epoch on concatenated SNLI+ANLI; validate on concatenated SNLI dev + ANLI dev; evaluate on SNLI and ANLI test rounds.

All runs use max_length=128 and default hyperparameters unless stated. Qualitative analysis inspects error types (negation, lexical overlap, world knowledge).

## 5 Results
**Table 1: Test accuracy (%)**

| Model (regime)            | SNLI | ANLI A1 | ANLI A2 | ANLI A3 | ANLI combined |
|---------------------------|-----:|--------:|--------:|--------:|--------------:|
| Baseline (SNLI-only)      | 89.29 |   30.70 |   30.40 |   30.83 |         30.64 |
| Control (SNLIâSNLI)       | 89.53 |   31.40 |   31.60 |   31.83 |         31.61 |
| Fixed (SNLIâSNLI+ANLI)    | 89.39 |   47.40 |   39.50 |   40.25 |         42.38 |

- In-domain (SNLI) performance (Table 1): All three models perform nearly identically on SNLI. The control model increases SNLI accuracy by only +0.24 points, indicating the baseline is near its ceiling. The fixed model, despite being trained on harder ANLI data, maintains virtually the same SNLI accuracy (â0.14 vs control). Adversarial fine-tuning does not harm in-domain performance; differences are not significant.
- Out-of-domain (ANLI) performance (Table 1): Baseline collapses on ANLI (â30.6%, essentially random for 3-way). Control is only +0.97 better. Fixed (SNLI+ANLI) improves dramatically: +16 on A1, +7.9 on A2, +8.42 on A3, +10.77 combined (+34% relative). Adversarial fine-tuning massively increases robustness on all rounds while preserving SNLI performance. Figure 1 visualizes the SNLI/ANLI trade-off, and Figure 2 breaks down ANLI improvements by round.

![Figure 1: SNLI vs ANLI accuracy across models](figs/image1.png)

![Figure 2: ANLI round-level accuracy comparison](figs/image2.png)

## 6 Error Analysis (qualitative)
Here are some errors present in Control evaluations but not in the Fixed evaluations (Table 2):

**Table 2: Qualitative errors fixed by ANLI fine-tuning**

| Premise (abridged)                                                     | Hypothesis                                           | Gold | Control pred | Error type        |
|------------------------------------------------------------------------|------------------------------------------------------|------|--------------|-------------------|
| World Premiere â¦ released by Paramount Pictures                        | World Premiere was not released by Universal Studios | E    | C            | Negation trap     |
| The Second Jungle Book â¦ not based on âThe Second Jungle Bookâ         | The first Jungle Book was written before 1997        | E    | C            | World knowledge   |
| Beat TV â¦ daily entertainment show with various celebrity guests       | The hosts on Beat TV shared the screen time equally  | N    | E            | Lexical overlap   |

E = entailment, N = neutral, C = contradiction.

## 7 Discussion and Limitations
- ANLI fine-tuning materially improves robustness while preserving SNLI accuracy, showing adversarial data helps without overfitting away in-domain performance.
- Gains are smaller on A2/A3 than A1, likely due to harder domains (news, fiction, spoken transcripts) and limited model capacity (ELECTRA-small).
- Threats to validity: single model size, single fine-tuning epoch for ANLI, no variance reported; results may be sensitive to learning rate/epochs.

## 8 Conclusion and Future Work
Brief adversarial fine-tuning (SNLIâSNLI+ANLI) boosts ANLI robustness by ~11 points with no SNLI cost. Future work: explore larger models, longer ANLI schedules, curriculum mixing of SNLI/ANLI, calibration or debiasing methods, and richer behavioral tests (contrast/checklist).
