# Abstract

Natural language inference (NLI) models based on pre-trained transformers can achieve high accuracy on benchmarks like SNLI, yet they often exhibit two intertwined failures: overconfident probabilities and brittle behavior on adversarial challenge sets. In this work we study two complementary fine-tuning strategies for ELECTRA-small: (1) a calibration-oriented contrastive learning objective that exploits naturally occurring premise–hypothesis bundles in SNLI, and (2) an adversarial robustness–oriented fine-tuning stage that mixes ANLI with SNLI. Our contrastive approach adds a margin ranking loss with lexical-overlap–based dynamic weighting on hypothesis pairs that share a premise but have different labels, encouraging the model to express uncertainty on fine-grained lexical and semantic distinctions. On SNLI, this contrastive fine-tuning leaves accuracy essentially unchanged (89.53%→89.58%) but dramatically improves calibration: Expected Calibration Error drops by 45.7% (0.066→0.036), Maximum Calibration Error by 41.6%, Negative Log-Likelihood by 13.8%, and the fraction of overconfident predictions (>0.99) from 58.3% to 32.5%, with extreme high-confidence errors reduced from 14.5% to 4.7% of all errors. On ANLI, it yields small but consistent accuracy gains on harder rounds (+0.7 on R2, +1.0 on R3). In a complementary setup, we perform a brief adversarial fine-tuning phase on concatenated SNLI+ANLI. Relative to a control model trained for an extra epoch on SNLI alone, this adversarial regime improves combined ANLI accuracy by +10.8 absolute points (≈34% relative) while maintaining SNLI accuracy (~89.4%), demonstrating substantial robustness gains without sacrificing in-domain performance. Taken together, our results show that calibration-aware contrastive fine-tuning and ANLI-based adversarial fine-tuning address distinct but synergistic aspects of NLI reliability: the former reshapes confidence distributions with minimal cost, while the latter substantially improves performance on challenging out-of-distribution data.

---

## 1. Introduction

Natural language inference (NLI) is a core benchmark for evaluating language understanding, requiring models to predict whether a hypothesis is entailed by, contradicts, or is neutral with respect to a premise (Bowman et al., 2015). Transformer-based models fine-tuned on SNLI routinely achieve high accuracy, but a growing body of work shows that this success often relies on superficial cues and dataset artifacts rather than robust reasoning. Hypothesis-only baselines, lexical overlap heuristics, and simple negation cues can explain much of the performance (Poliak et al., 2018; McCoy et al., 2019; Gardner et al., 2020), raising concerns about how well these models generalize beyond benchmark test sets.

Two reliability issues are particularly problematic in deployment:

1. **Overconfidence and poor calibration.** NLI models frequently assign near-certainty to incorrect predictions (Guo et al., 2017). In applications where predicted probabilities inform downstream decisions or risk assessments, miscalibration can be as harmful as low accuracy.
2. **Brittleness to adversarial examples.** Adversarial challenge sets such as ANLI (Nie et al., 2020) are constructed in a human-and-model-in-the-loop manner to expose model weaknesses through subtle perturbations (lexical substitutions, negation, world knowledge). Models that perform well on SNLI often collapse to near-random accuracy on ANLI.

Recent work has proposed several strategies to address these failures. To combat artifacts and shortcut learning, researchers have explored debiasing via ensembles (Clark et al., 2019; He et al., 2019), data augmentation and adversarial training (Liu et al., 2019; Morris et al., 2020), and instance reweighting based on dataset cartography (Swayamdipta et al., 2020) or confidence regularization (Utama et al., 2020). In parallel, calibration-specific approaches aim to align predicted probabilities with empirical correctness, using techniques such as temperature scaling (Guo et al., 2017) or calibration-aware training objectives. Contrastive learning, which encourages models to distinguish between similar but label-differing instances, has shown promise in reading comprehension and representation learning (Chen et al., 2020; Dua et al., 2021), but its calibration effects in NLI remain underexplored.

In this project we investigate two complementary fine-tuning strategies on a shared ELECTRA-small backbone (Clark et al., 2020) that directly target these reliability gaps:

- **A premise-bundled contrastive learning objective** on SNLI, which adds an overlap-weighted margin ranking loss over hypotheses sharing the same premise but having different labels. This objective is designed to improve calibration by forcing the model to express appropriate uncertainty on subtle lexical and semantic contrasts.
- **An adversarial fine-tuning regime** that mixes ANLI with SNLI for a brief additional epoch. This setup uses adversarially constructed ANLI examples to “correct” the SNLI-trained model, with the goal of improving performance on ANLI while preserving SNLI accuracy.

We frame these as two halves of a broader question: can we improve both calibration and adversarial robustness of an NLI model using lightweight, post-hoc fine-tuning strategies that minimally disturb its strong in-domain performance?

Our main contributions are:

1. **Contrastive calibration for NLI.** We propose a contrastive fine-tuning framework that leverages naturally occurring premise–hypothesis bundles in SNLI, with dynamic weights based on lexical overlap. We show that a single contrastive epoch yields large gains in calibration metrics with essentially unchanged SNLI accuracy.
2. **Adversarial robustness via ANLI fine-tuning.** We establish a strong SNLI ELECTRA-small baseline and compare an additional SNLI epoch (control) to a joint SNLI+ANLI fine-tuning regime. The adversarial run improves ANLI accuracy by +10.8 absolute points while preserving SNLI performance.
3. **Joint perspective on reliability.** By analyzing calibration and adversarial robustness together, we highlight how contrastive and adversarial fine-tuning tackle different failure modes: contrastive learning primarily reshapes confidence distributions, whereas ANLI fine-tuning primarily improves correctness on hard adversarial examples.
4. **Qualitative and categorical error analysis.** We present fine-grained error breakdowns across linguistic phenomena (negation, quantifiers, lexical overlap, world knowledge) and discuss where each strategy helps or fails, revealing complex redistributions of errors rather than uniform improvements.

---

## 2. Background and Related Work

### 2.1 Dataset Artifacts and Shortcut Learning in NLI

A central critique of NLI benchmarks is that they contain annotation artifacts—systematic correlations between surface patterns and labels that models can exploit without genuine reasoning. Poliak et al. (2018) show that hypothesis-only models achieve surprisingly strong performance on multiple NLI datasets, indicating that premise information is often unnecessary. McCoy et al. (2019) identify syntactic heuristics (e.g., “high lexical overlap implies entailment”) that models mistakenly internalize. Gardner et al. (2020) introduce contrast sets—minimally edited examples that flip the gold label—to demonstrate how brittle these shortcuts are under small perturbations.

To mitigate artifacts, several lines of work modify training data or objectives:

- **Ensemble and residual debiasing.** Clark et al. (2019) and He et al. (2019) train a “biased” model on spurious features and encourage a main model to focus on residual information.
- **Data augmentation and adversarial training.** Liu et al. (2019) propose “inoculation by fine-tuning”, adding targeted challenge examples to training. Morris et al. (2020) generate adversarial examples via meaning-preserving transformations to expose model weaknesses.
- **Instance reweighting and calibration-aware objectives.** Swayamdipta et al. (2020) introduce dataset cartography, identifying “hard” and “ambiguous” instances to guide reweighting. Utama et al. (2020) propose confidence regularization to prevent overreliance on biased cues.

Our work connects to this line by using premise-bundled contrastive pairs and adversarially constructed ANLI examples as targeted training signals against shortcut behavior.

### 2.2 Calibration of Neural Classifiers

A probabilistic classifier is well-calibrated when its confidence scores match empirical correctness frequencies (Guo et al., 2017). Modern deep networks tend to be overconfident, especially when trained with cross-entropy and aggressive regularization or data augmentation. Poor calibration is particularly problematic when outputs are consumed by downstream decision-making components, thresholding rules, or risk-sensitive systems.

Standard post-hoc calibration methods (e.g., temperature scaling) adjust predicted logits without changing the model’s decision boundaries. Training-time approaches instead modify objectives or architectures to encourage calibrated behavior, for example by adding regularizers that penalize miscalibrated confidence distributions. In NLI, however, most prior work has focused on accuracy and artifact mitigation rather than calibration metrics such as Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), or Brier score.

Our contrastive fine-tuning approach falls into this training-time calibration category. By explicitly contrasting hypotheses that share a premise but have different labels—and by weighting contrasts based on lexical overlap—we encourage the model to avoid unwarranted extreme confidence when distinctions are subtle.

### 2.3 ELECTRA and Adversarial Challenge Sets

ELECTRA (Clark et al., 2020) is a pre-trained transformer that replaces masked language modeling with a discriminative pre-training task: distinguishing real tokens from synthetic replacements. ELECTRA-small provides a strong, efficient encoder backbone for classification tasks like NLI; we use it as our shared base model in all experiments.

To assess genuine language understanding and robustness, several adversarial challenge datasets have been introduced. Among them, ANLI (Nie et al., 2020) is notable for its human-and-model-in-the-loop construction: annotators iteratively craft adversarial NLI examples that fool a current model. ANLI is released in three rounds (A1–A3) with different domains and increasing difficulty, including Wikipedia, news, fiction, spoken language, and instructional texts. Models trained only on SNLI often achieve near-random accuracy (≈33%) on ANLI, reflecting severe out-of-distribution (OOD) brittleness.

Prior work has shown that fine-tuning on adversarial data can improve robustness, but such gains may come at the expense of in-domain performance or may require complex training curricula. Our adversarial fine-tuning experiments adopt a simple regime—one additional epoch on concatenated SNLI+ANLI—to test how much robustness can be gained without harming SNLI accuracy.

### 2.4 Contrastive Learning in NLP

Contrastive learning has been highly successful in representation learning (Chen et al., 2020) and has recently been adapted to NLP tasks. In reading comprehension, Dua et al. (2021) introduce instance bundles—groups of examples sharing a context but with different answers—and apply contrastive objectives to encourage robust distinctions, showing improvements on adversarial evaluation sets.

Our contrastive NLI approach follows a similar bundle-based philosophy: we treat all hypotheses associated with a given SNLI premise as a bundle and impose margin-based ranking constraints on label scores for hypothesis pairs with different gold labels. Our main novelty lies in the lexical-overlap–based dynamic weighting, which prioritizes high-overlap contrasts thought to be especially challenging and calibration-relevant.

---

## 6. Conclusion

We studied two complementary fine-tuning strategies for improving the reliability of an ELECTRA-small NLI model trained on SNLI: premise-bundled contrastive fine-tuning for calibration and adversarial fine-tuning with ANLI for robustness.

On the calibration side, we introduced a contrastive objective that operates over SNLI premise–hypothesis bundles, adding a margin ranking loss between hypotheses with different labels and weighting these pairs by lexical overlap. A single epoch of contrastive fine-tuning left SNLI accuracy essentially unchanged (89.53%→89.58%) but dramatically improved calibration: ECE and MCE dropped by 45.7% and 41.6% respectively, NLL and Brier scores improved, and the fraction of extremely confident predictions (>0.99) fell from 58.3% to 32.5%. Crucially, the proportion of errors made with extreme confidence decreased from 14.5% to 4.7%, indicating that the model learned to express appropriate uncertainty when it was likely to be wrong. These gains came at negligible computational cost.

On the robustness side, we established a strong SNLI baseline and compared two continuation regimes: an additional epoch on SNLI alone (control) and an adversarial regime training on concatenated SNLI+ANLI. While all models maintained similar SNLI performance (~89.3–89.5%), the adversarial run improved ANLI performance dramatically: combined ANLI accuracy increased by +10.8 absolute points (≈34% relative) over the control model, with large gains across all ANLI rounds. This shows that adversarial fine-tuning can substantially boost OOD robustness without sacrificing in-domain performance, even with a short additional schedule.

Taken together, our results highlight that calibration and robustness are related but distinct objectives. Contrastive fine-tuning mainly reshapes the model’s confidence distribution, making its probabilities more trustworthy, while adversarial fine-tuning mainly improves correctness on challenging adversarial examples. Neither strategy alone solves all reliability issues, but they provide lightweight, complementary tools for making NLI models more dependable in practice.

---

## 7. Future Work

Our findings suggest several directions for extending and combining calibration- and robustness-oriented fine-tuning:

1. **Direct artifact and challenge-set testing.** We plan to evaluate both contrastive and adversarially fine-tuned models on targeted artifact benchmarks such as HANS and contrast sets, as well as hypothesis-only tests, to more directly quantify reductions in shortcut reliance.
2. **Multiple seeds and hyperparameter sweeps.** All current results are based on single-seed runs with fixed hyperparameters. Future work should explore variance across random seeds and investigate the sensitivity of both methods to the contrastive margin, overlap weighting function, mixing ratio of SNLI/ANLI, and number of fine-tuning epochs.
3. **Extended and interleaved training schedules.** Our contrastive and adversarial fine-tuning stages are brief (one epoch) and applied after standard SNLI training. It would be informative to study longer schedules, interleaving or alternating contrastive and adversarial batches, and curriculum strategies that gradually increase difficulty.
4. **Synthetic and structured contrast generation.** While SNLI and ANLI provide natural and adversarial contrasts, we could generate additional synthetic contrast pairs targeting specific linguistic phenomena such as negation, quantifiers, and world knowledge, then integrate them into the contrastive objective.
5. **Fine-grained calibration analysis.** Beyond global metrics like ECE and MCE, we aim to stratify calibration by overlap level, label type, and linguistic category, investigating, for example, whether high-overlap entailment decisions become better calibrated than low-overlap contradictions.
6. **Combining calibration and robustness objectives.** A natural next step is to jointly optimize for calibration and robustness, e.g., by applying overlap-weighted contrastive losses on both SNLI and ANLI examples or by adding calibration-aware regularizers during adversarial fine-tuning.
7. **Scaling to larger models and other datasets.** Finally, we would like to test whether our findings hold for larger ELECTRA variants or other architectures, and whether similar benefits arise on multi-genre NLI datasets such as MultiNLI, as well as in downstream applications that use NLI as a subcomponent.

---

## References

Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large 
annotated corpus for learning natural language inference. *EMNLP*.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework 
for contrastive learning of visual representations. *ICML*.

Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). Don't take the easy way 
out: Ensemble based methods for avoiding known dataset biases. *EMNLP*.

Clark, K., Luong, M. T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: 
Pre-training text encoders as discriminators rather than generators. *ICLR*.

Dua, D., Dasigi, P., Singh, S., & Gardner, M. (2021). Learning with instance 
bundles for reading comprehension. *EMNLP*.

Gardner, M., et al. (2020). Evaluating models' local decision boundaries via 
contrast sets. *Findings of EMNLP*.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of 
modern neural networks. *ICML*.

He, H., Zha, S., & Wang, H. (2019). Unlearn dataset bias in natural language 
inference by fitting the residual. *DeepLo Workshop*.

Liu, N. F., Schwartz, R., & Smith, N. A. (2019). Inoculation by fine-tuning: 
A method for analyzing challenge datasets. *NAACL*.

McCoy, T., Pavlick, E., & Linzen, T. (2019). Right for the wrong reasons: 
Diagnosing syntactic heuristics in natural language inference. *ACL*.

Morris, J. X., et al. (2020). TextAttack: A framework for adversarial 
attacks, data augmentation, and adversarial training in NLP. *EMNLP*.

Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., & Van Durme, B. 
(2018). Hypothesis only baselines in natural language inference. *SemEval*.

Swayamdipta, S., et al. (2020). Dataset cartography: Mapping and diagnosing 
datasets with training dynamics. *EMNLP*.

Utama, P. A., Moosavi, N. S., & Gurevych, I. (2020). Towards debiasing NLU 
models from unknown biases. *EMNLP*.

Clark et al. 2020. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.

Nie et al. 2020. Adversarial NLI: A new benchmark for natural language understanding.

Poliak et al. 2018; McCoy et al. 2019. Analyses of NLI artifacts and syntactic heuristics.

