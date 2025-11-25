import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple, Optional, Dict, Union, List, Any
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

QA_MAX_ANSWER_LENGTH = 30

#test chagnge
# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    output.predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics


class ContrastBundle:
    """Groups multiple hypothesis examples for the same premise."""
    def __init__(self, premise, hypotheses):
        """
        Args:
            premise: The shared premise text
            hypotheses: List of dicts with 'hypothesis' and 'label' keys
        """
        self.premise = premise
        self.hypotheses = hypotheses  # List of {'hypothesis': str, 'label': int}


class ContrastiveNLIDataset(Dataset):
    """Dataset that groups multiple hypotheses for the same premise."""

    def __init__(self, snli_dataset, tokenizer=None, max_length=128, min_hypotheses=2):
        """
        Args:
            snli_dataset: SNLI dataset (already filtered for label != -1)
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            min_hypotheses: Minimum number of hypotheses required per premise (default 2)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bundles = []

        # Group examples by premise
        from collections import defaultdict
        premise_groups = defaultdict(list)

        n_examples = len(snli_dataset)

        for idx in range(n_examples):
            ex = snli_dataset[idx]
            premise_groups[ex['premise']].append({
                'hypothesis': ex['hypothesis'],
                'label': ex['label']
            })

        # Create bundles from groups with enough hypotheses and diverse labels
        for premise, hypotheses in premise_groups.items():
            if len(hypotheses) >= min_hypotheses:
                # Check if we have at least 2 different labels for meaningful contrast
                unique_labels = set(h['label'] for h in hypotheses)
                if len(unique_labels) >= 2:
                    bundle = ContrastBundle(
                        premise=premise,
                        hypotheses=hypotheses
                    )
                    self.bundles.append(bundle)

        if self.bundles:
            print(f"Created {len(self.bundles)} bundles from {n_examples} examples")
            avg_hyp = sum(len(b.hypotheses) for b in self.bundles) / len(self.bundles)
            print(f"Average hypotheses per bundle: {avg_hyp:.1f}")

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        bundle = self.bundles[idx]

        # Tokenize all hypotheses with the same premise
        all_inputs = []
        all_labels = []

        for hyp_data in bundle.hypotheses:
            inputs = self.tokenizer(
                bundle.premise,
                hyp_data['hypothesis'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            all_inputs.append(inputs)
            all_labels.append(hyp_data['label'])

        # Stack all inputs and labels
        return {
            'input_ids': torch.stack([inp['input_ids'].squeeze() for inp in all_inputs]),
            'attention_mask': torch.stack([inp['attention_mask'].squeeze() for inp in all_inputs]),
            'labels': torch.tensor(all_labels),
            'num_hypotheses': len(bundle.hypotheses)
        }


class NLIContrastTrainer(Trainer):
    """
    Trainer that implements margin ranking loss for NLI.
    For bundles of hypotheses sharing the same premise, ensures that
    correct predictions have higher confidence than incorrect ones by a margin.
    """

    def __init__(self, *args, contrast_weight=0.5, margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast_weight = contrast_weight  # Weight for margin loss vs CE loss
        self.margin = margin  # Margin for ranking loss
        # Disable remove_unused_columns since we're using custom column names
        self.args.remove_unused_columns = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined cross-entropy and margin ranking loss.
        """
        # Check if this is bundled format
        if 'num_hypotheses' not in inputs:
            # Standard format - just use regular CE loss
            outputs = model(**inputs)
            loss = outputs.loss if outputs.loss is not None else outputs['loss']
            return (loss, outputs) if return_outputs else loss

        batch_size = inputs['input_ids'].size(0)
        # num_hypotheses should be the same for all items in a batch after padding
        num_hypotheses = inputs['input_ids'].size(1)  # Get from tensor shape instead

        # Reshape inputs to process all hypotheses
        # input_ids shape: (batch_size, num_hypotheses, seq_len)
        input_ids = inputs['input_ids'].view(batch_size * num_hypotheses, -1)
        attention_mask = inputs['attention_mask'].view(batch_size * num_hypotheses, -1)
        labels = inputs['labels'].view(-1)

        # Forward pass for all hypotheses
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits  # Shape: (batch_size * num_hypotheses, num_classes)

        # Standard cross-entropy loss (ignores -100 labels automatically)
        ce_loss = F.cross_entropy(logits, labels)

        # Compute margin ranking loss
        margin_losses = []

        # Process each bundle
        for b in range(batch_size):
            start_idx = b * num_hypotheses
            end_idx = start_idx + num_hypotheses

            bundle_logits = logits[start_idx:end_idx]  # (num_hypotheses, num_classes)
            bundle_labels = labels[start_idx:end_idx]   # (num_hypotheses,)

            # Skip padded examples (labeled with -100)
            valid_mask = bundle_labels != -100
            if valid_mask.sum() < 2:
                continue  # Need at least 2 valid examples for comparison

            valid_logits = bundle_logits[valid_mask]
            valid_labels = bundle_labels[valid_mask]

            # For each pair of hypotheses with different labels
            for i in range(len(valid_labels)):
                for j in range(i + 1, len(valid_labels)):
                    if valid_labels[i] != valid_labels[j]:
                        # Original ranking approach: each hypothesis should score higher 
                        # on its own true label than other hypotheses do
                        
                        # Get the label indices
                        label_i = valid_labels[i]
                        label_j = valid_labels[j]
                        
                        # Hypothesis i should score higher on label_i than hypothesis j does
                        score_i_on_label_i = valid_logits[i][label_i]
                        score_j_on_label_i = valid_logits[j][label_i]
                        loss_i = torch.relu(self.margin - (score_i_on_label_i - score_j_on_label_i))
                        margin_losses.append(loss_i)
                        
                        # Hypothesis j should score higher on label_j than hypothesis i does
                        score_j_on_label_j = valid_logits[j][label_j]
                        score_i_on_label_j = valid_logits[i][label_j]
                        loss_j = torch.relu(self.margin - (score_j_on_label_j - score_i_on_label_j))
                        margin_losses.append(loss_j)

        # Combine losses
        if margin_losses:
            margin_loss = torch.stack(margin_losses).mean()
            total_loss = (1 - self.contrast_weight) * ce_loss + self.contrast_weight * margin_loss
        else:
            total_loss = ce_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs before sending them to the model.
        """
        # Move inputs to the correct device
        inputs = self._prepare_input(inputs)

        # Handle bundled inputs
        for key in ['input_ids', 'attention_mask', 'labels', 'num_hypotheses']:
            if key in inputs and isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.args.device)

        return inputs


class ContrastiveDataCollator:
    """Custom data collator for contrastive learning that handles bundled examples."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        if not features:
            return {}

        # Check if this is bundled format with num_hypotheses
        if isinstance(features[0], dict) and 'num_hypotheses' in features[0]:
            # Find max number of hypotheses for padding
            max_hypotheses = max(f['num_hypotheses'] for f in features)
            seq_len = features[0]['input_ids'].size(-1)

            batch = {
                'input_ids': [],
                'attention_mask': [],
                'labels': [],
                'num_hypotheses': []
            }

            for feature in features:
                num_hyp = feature['num_hypotheses']

                # Pad if needed
                if num_hyp < max_hypotheses:
                    pad_size = max_hypotheses - num_hyp
                    # Pad with zeros
                    padded_input_ids = torch.cat([
                        feature['input_ids'],
                        torch.zeros(pad_size, seq_len, dtype=feature['input_ids'].dtype)
                    ])
                    padded_attention_mask = torch.cat([
                        feature['attention_mask'],
                        torch.zeros(pad_size, seq_len, dtype=feature['attention_mask'].dtype)
                    ])
                    # For labels, pad with -100 (ignored by loss)
                    padded_labels = torch.cat([
                        feature['labels'],
                        torch.full((pad_size,), -100, dtype=feature['labels'].dtype)
                    ])
                else:
                    padded_input_ids = feature['input_ids']
                    padded_attention_mask = feature['attention_mask']
                    padded_labels = feature['labels']

                batch['input_ids'].append(padded_input_ids)
                batch['attention_mask'].append(padded_attention_mask)
                batch['labels'].append(padded_labels)
                batch['num_hypotheses'].append(num_hyp)

            # Stack into batch tensors
            batch['input_ids'] = torch.stack(batch['input_ids'])
            batch['attention_mask'] = torch.stack(batch['attention_mask'])
            batch['labels'] = torch.stack(batch['labels'])
            batch['num_hypotheses'] = torch.tensor(batch['num_hypotheses'])

            return batch
        else:
            # Standard format - use default collator
            from transformers import default_data_collator
            return default_data_collator(features)
