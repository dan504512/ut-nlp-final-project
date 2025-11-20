# Research Paper Summaries for NLP Final Project

This document contains summaries of all papers referenced in the CS388 NLP Final Project, organized chronologically. Each summary includes the paper's key contribution, why it's worth reading for this project, and core concepts.

---

## 2015

### A Large Annotated Corpus for Learning Natural Language Inference
**Authors:** Bowman, Angeli, Potts, Manning  
**Paper:** bowman2015_snli.pdf

**Key Contribution:**  
Introduces SNLI (Stanford Natural Language Inference), a corpus of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral. This was the first large-scale corpus for NLI that enabled training data-hungry neural models.

**Why Read:**  
- Foundational dataset for studying NLI artifacts
- Describes annotation process that inadvertently introduces biases
- Baseline experiments reveal patterns that models exploit

**Core Concepts:**  
- **Natural Language Inference (NLI)**: Task of determining whether a hypothesis is true (entailment), false (contradiction), or undetermined (neutral) given a premise
- **Crowdsourced annotation**: Using Amazon Mechanical Turk workers to create hypotheses given premises
- **Annotation artifacts**: Unintended patterns in data creation (e.g., negation words correlate with contradiction)

---

## 2016

### SQuAD: 100,000+ Questions for Machine Comprehension of Text
**Authors:** Rajpurkar, Zhang, Lopyrev, Liang  
**Paper:** rajpurkar2016_squad.pdf

**Key Contribution:**  
Introduces SQuAD (Stanford Question Answering Dataset), containing 100,000+ questions posed by crowdworkers on Wikipedia articles where the answer is a segment of text from the reading passage. Revolutionized reading comprehension research.

**Why Read:**  
- Another major dataset prone to artifacts and shortcuts
- Discusses evaluation metrics and their limitations
- Shows how models can achieve high scores without true comprehension

**Core Concepts:**  
- **Extractive QA**: Answers must be extracted as spans from the context
- **Crowdsourced questions**: Workers asked to pose questions about Wikipedia paragraphs
- **Lexical overlap**: Models exploit word matching between question and context

---

## 2017

### Adversarial Examples for Evaluating Reading Comprehension Systems
**Authors:** Jia, Liang  
**Paper:** jia2017_adversarial_squad.pdf

**Key Contribution:**  
Introduces AddSent and AddOneSent methods to generate adversarial examples for reading comprehension by adding distracting sentences to paragraphs. Shows that state-of-the-art models fail catastrophically on these examples despite humans being unaffected.

**Why Read:**  
- Pioneering work on adversarial evaluation for QA systems
- Simple but effective method for exposing model brittleness
- Template for creating challenging evaluation sets

**Core Concepts:**  
- **Adversarial examples**: Inputs designed to fool models while preserving human performance
- **Distractor sentences**: Sentences that contain question words but don't answer the question
- **Overstability**: Model's inability to change predictions when semantically significant changes are made

---

## 2018

### A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference
**Authors:** Williams, Nangia, Bowman  
**Paper:** williams2018_multinli.pdf

**Key Contribution:**  
Introduces MultiNLI, a corpus of 433k sentence pairs covering a range of genres of spoken and written text. Extends SNLI with greater linguistic diversity and cross-genre evaluation capabilities.

**Why Read:**  
- Shows how artifacts persist across different data collection efforts
- Genre diversity reveals model generalization failures
- Matched vs. mismatched evaluation exposes overfitting patterns

**Core Concepts:**  
- **Genre diversity**: Telephone speech, fiction, government reports, etc.
- **Matched/Mismatched evaluation**: Testing on same vs. different genres as training
- **Cross-genre transfer**: Models struggle when genre shifts despite task being the same

### HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering
**Authors:** Yang et al.  
**Paper:** yang2018_hotpotqa.pdf

**Key Contribution:**  
Introduces HotpotQA with 113k Wikipedia-based question-answer pairs requiring reasoning over multiple documents. Features diverse reasoning types and supporting facts annotations for explainability.

**Why Read:**  
- Multi-hop reasoning creates new types of exploitable patterns
- Supporting facts annotations reveal shortcut learning
- Comparison questions show systematic biases

**Core Concepts:**  
- **Multi-hop reasoning**: Questions requiring information from multiple paragraphs
- **Supporting facts**: Sentence-level annotations showing evidence for answers
- **Bridge entities**: Entities connecting reasoning chains that models learn to exploit

### Hypothesis Only Baselines in Natural Language Inference
**Authors:** Poliak et al.  
**Paper:** poliak2018_hypothesis_only.pdf

**Key Contribution:**  
Demonstrates that models trained only on hypotheses (without premises) can achieve surprisingly high accuracy on NLI tasks, revealing severe annotation artifacts in multiple NLI datasets.

**Why Read:**  
- Shocking evidence of dataset artifacts in NLI
- Simple diagnostic technique you can apply
- Quantifies the severity of the bias problem

**Core Concepts:**  
- **Hypothesis-only baseline**: Training models without access to premises
- **Annotation artifacts**: Statistical patterns that leak label information
- **Dataset-specific biases**: Different datasets have different exploitable patterns

### Breaking NLI Systems with Sentences that Require Simple Lexical Inferences
**Authors:** Glockner, Shwartz, Goldberg  
**Paper:** glockner2018_breaking_nli.pdf

**Key Contribution:**  
Creates a challenging test set by replacing single words in SNLI with synonyms, antonyms, or hypernyms. State-of-the-art models catastrophically fail on these simple modifications.

**Why Read:**  
- Simple method for generating hard examples
- Reveals models' lack of lexical understanding
- Shows the gap between benchmark performance and linguistic competence

**Core Concepts:**  
- **Lexical inference**: Understanding word relationships (synonymy, antonymy, hypernymy)
- **Minimal pairs**: Examples differing by single word changes
- **Catastrophic failure**: Models performing near chance on seemingly simple modifications

### How Much Reading Does Reading Comprehension Require?
**Authors:** Kaushik, Lipton  
**Paper:** kaushik2018_reading_comprehension.pdf

**Key Contribution:**  
Shows that competitive performance on reading comprehension benchmarks can be achieved using simple models that ignore the passage, revealing that many questions can be answered using prior knowledge or question patterns alone.

**Why Read:**  
- Challenges assumptions about what models learn
- Proposes diagnostic ablations for QA systems
- Quantifies information sources needed for different questions

**Core Concepts:**  
- **Passage-independent answering**: Answering without reading the context
- **Type-matching heuristics**: Matching answer types to question types
- **Information attribution**: Determining if answers come from passage vs. prior knowledge

---

## 2019

### Understanding Dataset Design Choices for Multi-hop Reasoning
**Authors:** Chen, Durrett  
**Paper:** chen2019_multihop_reasoning.pdf

**Key Contribution:**  
Analyzes how dataset design choices in multi-hop QA affect what models learn. Shows that models often use shortcuts rather than performing multi-hop reasoning, and proposes methods to diagnose these issues.

**Why Read:**  
- Framework for analyzing reasoning requirements
- Diagnostic techniques for multi-hop datasets
- Shows how construction methods create artifacts

**Core Concepts:**  
- **Reasoning shortcuts**: Single-hop solutions to multi-hop problems
- **Decomposition analysis**: Breaking complex questions into sub-questions
- **Answerability**: Whether partial information suffices for correct answers

### Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases
**Authors:** Clark, Yatskar, Zettlemoyer  
**Paper:** clark2019_dont_take_easy_way.pdf

**Key Contribution:**  
Proposes ensemble-based debiasing where a biased model learns superficial patterns, then the main model is trained to learn the residual. Shows improvements on challenging evaluation sets.

**Why Read:**  
- Practical debiasing technique you can implement
- Theoretical framework for bias removal
- Strong empirical results on multiple datasets

**Core Concepts:**  
- **Product of Experts (PoE)**: Combining biased and debiased model predictions
- **Bias-only model**: Weak model capturing superficial correlations
- **Residual learning**: Main model learns patterns not captured by bias model

### Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual
**Authors:** He, Zha, Wang  
**Paper:** he2019_unlearn_dataset_bias.pdf

**Key Contribution:**  
Proposes training models to predict the residual errors of a biased model, effectively forcing the model to learn patterns beyond superficial correlations.

**Why Read:**  
- Alternative approach to debiasing
- Mathematical framework for bias removal
- Connections to causal inference

**Core Concepts:**  
- **Residual fitting**: Learning to correct biased model errors
- **Z-filtering**: Removing examples well-predicted by biased model
- **Debiased training objective**: Modified loss focusing on hard examples

### Inoculation by Fine-tuning: A Method for Analyzing Challenge Datasets
**Authors:** Liu, Schwartz, Smith  
**Paper:** liu2019_inoculation.pdf

**Key Contribution:**  
Introduces "inoculation" - fine-tuning on small amounts of challenge data to improve robustness. Shows this simple method can substantially improve performance on adversarial datasets.

**Why Read:**  
- Simple, practical improvement technique
- Analysis of how much challenge data is needed
- Trade-offs between in-domain and challenge performance

**Core Concepts:**  
- **Inoculation**: Exposing models to small amounts of challenging data
- **Challenge datasets**: Adversarially constructed evaluation sets
- **Fine-tuning strategies**: How to incorporate challenge data effectively

### Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference
**Authors:** McCoy, Pavlick, Linzen  
**Paper:** mccoy2019_right_wrong_reasons.pdf

**Key Contribution:**  
Identifies three syntactic heuristics that BERT uses for NLI (lexical overlap, subsequence, constituent). Creates HANS dataset to test these specific heuristics.

**Why Read:**  
- Systematic framework for identifying heuristics
- Targeted evaluation set construction
- Clear examples of what models actually learn

**Core Concepts:**  
- **Syntactic heuristics**: Shallow patterns based on word order and structure
- **Lexical overlap heuristic**: Assuming entailment when words overlap
- **Constituent heuristic**: Assuming entailment for embedded constituents
- **HANS dataset**: Targeted test for specific heuristics

### Universal Adversarial Triggers for Attacking and Analyzing NLP
**Authors:** Wallace et al.  
**Paper:** wallace2019_universal_triggers.pdf

**Key Contribution:**  
Discovers universal adversarial triggers - tokens that, when concatenated to any input, cause targeted model failures. Shows these triggers transfer across examples and sometimes across models.

**Why Read:**  
- Reveals fundamental model vulnerabilities
- Method for stress-testing any NLP model
- Insights into what models actually learn

**Core Concepts:**  
- **Universal triggers**: Input-agnostic adversarial tokens
- **Gradient-based search**: Finding triggers via optimization
- **Trigger transferability**: Triggers working across different inputs/models

---

## 2020

### ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
**Authors:** Clark, Luong, Le, Manning  
**Paper:** clark2020_electra.pdf

**Key Contribution:**  
Proposes ELECTRA, a pre-training method that trains models to distinguish real tokens from fake ones rather than masked language modeling. More sample-efficient than BERT, especially for smaller models.

**Why Read:**  
- Recommended base model for the project
- More efficient than BERT for small-scale experiments
- Better performance with limited compute

**Core Concepts:**  
- **Replaced token detection**: Discriminating real vs. generated tokens
- **Generator-discriminator setup**: Small generator creates corruptions
- **Sample efficiency**: Better performance with less pre-training data

### Evaluating Models' Local Decision Boundaries via Contrast Sets
**Authors:** Gardner et al.  
**Paper:** gardner2020_contrast_sets.pdf

**Key Contribution:**  
Introduces contrast sets - expert-crafted perturbations that change gold labels. Shows that models are brittle to these meaningful perturbations despite high original accuracy.

**Why Read:**  
- Gold standard for evaluation set construction
- Framework for analyzing model robustness
- Templates for creating your own contrast sets

**Core Concepts:**  
- **Contrast sets**: Small perturbations that flip the correct label
- **Local decision boundaries**: How models behave near specific examples
- **Expert annotation**: Using human expertise to create meaningful perturbations
- **Consistency**: Whether models update predictions appropriately

### Beyond Accuracy: Behavioral Testing of NLP Models with CheckList
**Authors:** Ribeiro, Wu, Guestrin, Singh  
**Paper:** ribeiro2020_checklist.pdf

**Key Contribution:**  
Introduces CheckList, a task-agnostic methodology for testing NLP models using templates, lexicons, and perturbations. Reveals failures in commercial and research systems.

**Why Read:**  
- Comprehensive testing framework you can apply
- Systematic approach to finding model failures
- Tools for generating test cases at scale

**Core Concepts:**  
- **Minimum Functionality Tests (MFT)**: Testing specific capabilities
- **Invariance Tests (INV)**: Changes that shouldn't affect predictions
- **Directional Expectation Tests (DIR)**: Changes with predictable effects
- **Template-based generation**: Scaling test creation

### Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics
**Authors:** Swayamdipta et al.  
**Paper:** swayamdipta2020_dataset_cartography.pdf

**Key Contribution:**  
Introduces dataset cartography - using training dynamics to categorize examples as easy, hard, or ambiguous. Shows these categories reveal dataset quality issues and can improve training.

**Why Read:**  
- Main example in the project description
- Practical tool for dataset analysis
- Framework for selective training

**Core Concepts:**  
- **Training dynamics**: How predictions evolve during training
- **Confidence**: Mean model confidence across epochs
- **Variability**: Variance in model confidence across epochs
- **Data maps**: 2D visualization of dataset regions

### Towards Robustifying NLI Models Against Lexical Dataset Biases
**Authors:** Zhou, Bansal  
**Paper:** zhou2020_robustifying_nli.pdf

**Key Contribution:**  
Proposes several debiasing strategies including example reweighting and adversarial training. Shows combining methods improves robustness to lexical biases.

**Why Read:**  
- Comparison of multiple debiasing approaches
- Practical implementation details
- Analysis of method complementarity

**Core Concepts:**  
- **Lexical bias**: Word-level statistical correlations
- **Example reweighting**: Adjusting importance based on bias
- **Adversarial debiasing**: Training to fool a bias-only model

### Towards Debiasing NLU Models from Unknown Biases
**Authors:** Utama, Moosavi, Gurevych  
**Paper:** utama2020_debiasing_nlu.pdf

**Key Contribution:**  
Proposes self-debiasing without knowing biases a priori. Uses shallow models to identify biased examples automatically, then adjusts training accordingly.

**Why Read:**  
- Bias-agnostic debiasing approach
- Practical when biases are unknown
- Generalizable across tasks

**Core Concepts:**  
- **Unknown biases**: Artifacts not identified beforehand
- **Shallow model ensemble**: Multiple weak learners finding different biases
- **Confidence-based reweighting**: Down-weighting examples shallow models get right

### What Can We Learn from Collective Human Opinions on Natural Language Inference Data?
**Authors:** Nie, Zhou, Bansal  
**Paper:** nie2020_collective_opinions.pdf

**Key Contribution:**  
Collects multiple annotations per example to study human disagreement. Shows that considering annotation distribution rather than single labels reveals dataset issues and improves models.

**Why Read:**  
- Rethinking "ground truth" in NLP
- Human disagreement as signal not noise
- Alternative training objectives

**Core Concepts:**  
- **Annotation distribution**: Full distribution of human labels
- **Inherent ambiguity**: Some examples have legitimate disagreement
- **Soft labels**: Training with distributions instead of hard labels

### TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training
**Authors:** Morris et al.  
**Paper:** morris2020_textattack.pdf

**Key Contribution:**  
Introduces TextAttack, a Python framework for adversarial attacks in NLP. Provides standardized implementations of 16 attacks and tools for robustness evaluation.

**Why Read:**  
- Practical toolkit you can use
- Comprehensive attack taxonomy
- Standardized evaluation protocols

**Core Concepts:**  
- **Attack recipes**: Combinations of transformations and constraints
- **Search methods**: Strategies for finding adversarial examples
- **Semantic constraints**: Ensuring perturbations preserve meaning

### Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension
**Authors:** Bartolo et al.  
**Paper:** bartolo2020_beat_the_ai.pdf

**Key Contribution:**  
Creates adversarial SQuAD examples through human-model collaboration where annotators try to fool models. Shows models fail on these examples while humans maintain performance.

**Why Read:**  
- Human-in-the-loop adversarial generation
- Analysis of what makes questions hard
- Iterative dataset improvement

**Core Concepts:**  
- **Adversarial annotation**: Humans deliberately creating hard examples
- **Model-in-the-loop**: Real-time model feedback during annotation
- **Validated adversarial examples**: Ensuring human solvability

---

## 2021

### Competency Problems: On Finding and Removing Artifacts in Language Data
**Authors:** Gardner et al.  
**Paper:** gardner2021_competency_problems.pdf

**Key Contribution:**  
Proposes automatic methods for finding artifacts using n-gram statistics. Shows simple patterns often suffice for high accuracy, questioning what models really learn.

**Why Read:**  
- Automated artifact detection
- Statistical framework for finding biases
- Applicable to any classification dataset

**Core Concepts:**  
- **Competency problems**: Artifacts allowing success without competence
- **Pointwise mutual information (PMI)**: Measuring feature-label correlation
- **N-gram artifacts**: Simple word patterns predicting labels
- **AFLite algorithm**: Automated artifact finding

### Learning with Instance Bundles for Reading Comprehension
**Authors:** Dua et al.  
**Paper:** dua2021_instance_bundles.pdf

**Key Contribution:**  
Proposes training with instance bundles - groups of related questions about the same passage. Shows this improves robustness and reduces reliance on shortcuts.

**Why Read:**  
- Novel training paradigm
- Addresses question-specific biases
- Practical improvement method

**Core Concepts:**  
- **Instance bundles**: Multiple questions per context
- **Contrastive learning**: Learning from related examples
- **Question diversity**: Forcing comprehensive passage understanding

### Learning from Others' Mistakes: Avoiding Dataset Biases without Modeling Them
**Authors:** Sanh, Wolf, Belinkov, Rush  
**Paper:** sanh2021_learning_from_mistakes.pdf

**Key Contribution:**  
Proposes PoE (Product of Experts) training where models learn to avoid biases captured by weak baselines without explicitly modeling those biases.

**Why Read:**  
- Elegant debiasing approach
- No need to identify specific biases
- Strong theoretical foundation

**Core Concepts:**  
- **Product of Experts**: Multiplicative model combination
- **Weak expert**: Simple model capturing biases
- **Orthogonal learning**: Main model learns complementary patterns

### Embracing Ambiguity: Shifting the Training Target of NLI Models
**Authors:** Meissner et al.  
**Paper:** meissner2021_embracing_ambiguity.pdf

**Key Contribution:**  
Proposes training NLI models on annotation distributions rather than gold labels, embracing inherent ambiguity in the task. Shows improved calibration and performance.

**Why Read:**  
- Rethinking NLI task formulation
- Handling legitimate disagreement
- Better uncertainty quantification

**Core Concepts:**  
- **Distributional targets**: Training on label distributions
- **Annotator disagreement**: Signal of inherent ambiguity
- **Calibrated predictions**: Probabilities reflecting true uncertainty

### Increasing Robustness to Spurious Correlations using Forgettable Examples
**Authors:** Yaghoobzadeh et al.  
**Paper:** yaghoobzadeh2021_forgettable_examples.pdf

**Key Contribution:**  
Identifies "forgettable" examples (learned then forgotten during training) as indicators of real patterns vs. memorization. Uses this signal to improve robustness.

**Why Read:**  
- Novel perspective on training dynamics
- Automatic identification of important examples
- Connection to curriculum learning

**Core Concepts:**  
- **Forgettable examples**: Examples with inconsistent training dynamics
- **Memorized examples**: Consistently easy examples (likely artifacts)
- **Example forgetting events**: Transitions from correct to incorrect predictions
- **Robust training**: Upweighting forgettable examples

---

## Summary and Connections

These papers form a comprehensive view of the dataset artifacts problem in NLP:

1. **Problem Identification** (2015-2018): Early work identifying artifacts in major datasets
2. **Diagnostic Methods** (2018-2019): Systematic approaches to finding and measuring artifacts
3. **Debiasing Techniques** (2019-2021): Various methods to mitigate artifacts
4. **Evaluation Frameworks** (2020-2021): Comprehensive testing and analysis tools

Key themes across papers:
- Models exploit superficial patterns rather than learning intended tasks
- Simple baselines often reveal these issues
- Multiple complementary approaches exist for mitigation
- Evaluation beyond accuracy is crucial
- Human judgment and ambiguity matter

These papers provide the theoretical foundation and practical tools needed for the final project's goal of analyzing and mitigating dataset artifacts.