import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

import numpy as np
from collections import Counter
from datasets import concatenate_datasets, Features, Value


NUM_PREPROCESSING_WORKERS = 2


def convert_contrast_labels(examples):
    """Convert SNLI contrast dataset labels from positive/negative to numeric NLI labels.

    The contrast dataset uses 'positive'/'negative' labels with varying instructions.
    When instruction mentions "logically inferred", positive means entailment.
    Otherwise, positive means non-entailment (contradiction).
    """
    labels = []
    for instruction, label_name in zip(examples['instruction'], examples['label_name']):
        if "logically inferred" in instruction:
            # Instruction says hypothesis is entailed from premise
            labels.append(0 if label_name == 'positive' else 2)  # 0=entailment, 2=contradiction
        else:
            # Instruction says hypothesis contradicts/is unrelated
            labels.append(2 if label_name == 'positive' else 0)  # 2=contradiction, 0=entailment
    examples['label'] = labels
    return examples


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.
        Special values: 'snli+contrast' combines SNLI with contrast examples (preserves neutral labels),
        'snli-contrast' uses only the contrast dataset (no neutral labels).""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--do_eval_anli', action='store_true',
                      help='Evaluate the model on the ANLI (Adversarial NLI) dataset.')
    argp.add_argument('--do_eval_contrast', action='store_true',
                      help='Evaluate the model on the NLI contrast sets.')
    argp.add_argument('--print_confusion_matrix', type=str, default=None,
                      help='Print confusion matrix for the specified predictions file (JSONL format).')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    elif args.dataset == 'snli+contrast':
        # Combined SNLI and SNLI contrast dataset for training
        dataset_id = ('snli+contrast',)
        print("Loading combined SNLI + SNLI contrast dataset...")

        # Load both datasets
        snli_dataset = datasets.load_dataset('snli')
        contrast_dataset = datasets.load_dataset('AntoineBlanot/snli-contrast')

        # Process contrast dataset using shared label conversion function
        contrast_train = contrast_dataset['train'].map(convert_contrast_labels, batched=True) if 'train' in contrast_dataset else None
        contrast_test = contrast_dataset['test'].map(convert_contrast_labels, batched=True)

        # Filter SNLI to remove examples with no label (-1)
        snli_train = snli_dataset['train'].filter(lambda ex: ex['label'] != -1)
        snli_val = snli_dataset['validation'].filter(lambda ex: ex['label'] != -1)

        # Remove extra columns from contrast dataset first
        if contrast_train:
            contrast_train = contrast_train.remove_columns(['instruction', 'label_name'])
        contrast_test = contrast_test.remove_columns(['instruction', 'label_name'])

        # Cast both datasets to have the same feature types
        common_features = Features({
            'premise': Value('string'),
            'hypothesis': Value('string'),
            'label': Value('int64')
        })

        snli_train = snli_train.cast(common_features)
        snli_val = snli_val.cast(common_features)
        if contrast_train:
            contrast_train = contrast_train.cast(common_features)
        contrast_test = contrast_test.cast(common_features)

        # Combine datasets
        dataset = {}
        if contrast_train:
            # Combine SNLI train with contrast train
            dataset['train'] = concatenate_datasets([snli_train, contrast_train])
            print(f"Combined training set: {len(snli_train)} SNLI + {len(contrast_train)} contrast = {len(dataset['train'])} examples")
        else:
            # Use only SNLI train if no contrast train available
            dataset['train'] = snli_train
            print(f"Training set: {len(dataset['train'])} SNLI examples (no contrast train split available)")

        # Use SNLI validation for evaluation
        dataset['validation'] = snli_val
        eval_split = 'validation'

    elif args.dataset == 'snli-contrast':
        # Special handling for SNLI contrast dataset
        dataset_id = ('AntoineBlanot/snli-contrast',)
        print("Loading SNLI contrast dataset for training...")
        raw_dataset = datasets.load_dataset('AntoineBlanot/snli-contrast')

        # Apply label conversion to both train and test splits using shared function
        dataset = {}
        if 'train' in raw_dataset:
            dataset['train'] = raw_dataset['train'].map(convert_contrast_labels, batched=True)
        dataset['test'] = raw_dataset['test'].map(convert_contrast_labels, batched=True)

        eval_split = 'test'
        print(f"SNLI contrast dataset loaded: {len(dataset.get('train', []))} train, {len(dataset['test'])} test examples")
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)

    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    anli_datasets_featurized = {}
    contrast_datasets_featurized = {}
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval or args.do_eval_anli or args.do_eval_contrast:
        # Handle ANLI evaluation
        if args.do_eval_anli:
            # Load ANLI dataset
            anli_dataset = datasets.load_dataset('anli')
            # ANLI has three rounds: r1, r2, r3
            for round_name in ['test_r1', 'test_r2', 'test_r3']:
                anli_round = anli_dataset[round_name]
                if args.max_eval_samples:
                    anli_round = anli_round.select(range(min(args.max_eval_samples, len(anli_round))))
                anli_datasets_featurized[round_name] = anli_round.map(
                    prepare_eval_dataset,
                    batched=True,
                    num_proc=NUM_PREPROCESSING_WORKERS,
                    remove_columns=anli_round.column_names
                )

        # Handle contrast set evaluation
        if args.do_eval_contrast:
            # Load SNLI contrast set
            try:
                print("Loading SNLI contrast set...")
                # Load SNLI contrast set from AntoineBlanot
                contrast_dataset = datasets.load_dataset('AntoineBlanot/snli-contrast')

                contrast_data = contrast_dataset['test']

                # Convert label_name to numeric label using shared function
                contrast_data = contrast_data.map(convert_contrast_labels, batched=True)

                if args.max_eval_samples:
                    contrast_data = contrast_data.select(range(min(args.max_eval_samples, len(contrast_data))))

                contrast_datasets_featurized['snli_contrast'] = contrast_data.map(
                    prepare_eval_dataset,
                    batched=True,
                    num_proc=NUM_PREPROCESSING_WORKERS,
                    remove_columns=contrast_data.column_names
                )
                print(f"Successfully loaded SNLI contrast set with {len(contrast_data)} examples")
            except Exception as e:
                print(f"Warning: Could not load SNLI contrast set: {e}")
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy


    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
        
        #put wrong predictions in a separate file
        with open(os.path.join(training_args.output_dir, 'eval_wrong_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'nli':
                for i, example in enumerate(eval_dataset):
                    predicted_label = int(eval_predictions.predictions[i].argmax())
                    if predicted_label != example['label']:
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = predicted_label
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')

        #print and save confusion matrix for NLI task
        if args.task == 'nli':
            import numpy as np
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            true_labels = [example['label'] for example in eval_dataset]
            predicted_labels = [int(pred.argmax()) for pred in eval_predictions.predictions]

            cm = confusion_matrix(true_labels, predicted_labels)
            print('Confusion Matrix:')
            print(cm)

            # Save confusion matrix as image
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                        yticklabels=['Entailment', 'Neutral', 'Contradiction'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(training_args.output_dir, 'confusion_matrix_SNLI.png'))
            plt.close()

    # Evaluate on ANLI if requested
    if args.do_eval_anli:
        print('\nEvaluating on ANLI dataset...')
        anli_results = {}
        anli_predictions = {}

        # Load the original ANLI dataset for saving predictions with examples
        anli_dataset_original = datasets.load_dataset('anli')

        for round_name in ['test_r1', 'test_r2', 'test_r3']:
            if round_name in anli_datasets_featurized:
                print(f'\nEvaluating on ANLI {round_name}...')
                # Reset eval_predictions for this round
                eval_predictions = None

                # Update trainer's eval dataset
                trainer.eval_dataset = anli_datasets_featurized[round_name]
                round_results = trainer.evaluate()
                anli_results[round_name] = round_results
                anli_predictions[round_name] = eval_predictions
                print(f'ANLI {round_name} results:')
                print(round_results)

        # Save ANLI results
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'anli_eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(anli_results, f, indent=2)
        
        # save predictions on ANLI rounds including premise and hypothesis text
        with open(os.path.join(training_args.output_dir, 'anli_eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for round_name in ['test_r1', 'test_r2', 'test_r3']:
                if round_name in anli_datasets_featurized:
                    trainer.eval_dataset = anli_datasets_featurized[round_name]
                    eval_dataset = anli_datasets_featurized[round_name]
                    eval_predictions = trainer.predict(eval_dataset)
                    # To get the original text, we need to refer back to the raw ANLI dataset
                    raw_anli_round = datasets.load_dataset('anli', split=round_name)
                    for i, example in enumerate(eval_dataset):
                        original_example = raw_anli_round[i]
                        example_with_prediction = dict(example)
                        example_with_prediction['premise'] = original_example['premise']
                        example_with_prediction['hypothesis'] = original_example['hypothesis']
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                        example_with_prediction['round'] = round_name
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')

        # save wrong predictions on ANLI rounds including premise and hypothesis text
        with open(os.path.join(training_args.output_dir, 'anli_eval_wrong_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for round_name in ['test_r1', 'test_r2', 'test_r3']:
                if round_name in anli_datasets_featurized:
                    trainer.eval_dataset = anli_datasets_featurized[round_name]
                    eval_dataset = anli_datasets_featurized[round_name]
                    eval_predictions = trainer.predict(eval_dataset)
                    # To get the original text, we need to refer back to the raw ANLI dataset
                    raw_anli_round = datasets.load_dataset('anli', split=round_name)
                    for i, example in enumerate(eval_dataset):
                        predicted_label = int(eval_predictions.predictions[i].argmax())
                        if predicted_label != example['label']:
                            original_example = raw_anli_round[i]
                            example_with_prediction = dict(example)
                            example_with_prediction['premise'] = original_example['premise']
                            example_with_prediction['hypothesis'] = original_example['hypothesis']
                            example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                            example_with_prediction['predicted_label'] = predicted_label
                            example_with_prediction['round'] = round_name
                            f.write(json.dumps(example_with_prediction))
                            f.write('\n')

        # Save ANLI predictions
        for round_name, preds in anli_predictions.items():
            if preds is not None:
                with open(os.path.join(training_args.output_dir, f'anli_{round_name}_predictions.jsonl'), encoding='utf-8', mode='w') as f:
                    original_data = anli_dataset_original[round_name]
                    if args.max_eval_samples:
                        original_data = original_data.select(range(min(args.max_eval_samples, len(original_data))))

                    for i, example in enumerate(original_data):
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = preds.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(preds.predictions[i].argmax())
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
        # Save combined ANLI predictions
        with open(os.path.join(training_args.output_dir, f'anli_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for round_name, preds in anli_predictions.items():
                if preds is not None:
                    original_data = anli_dataset_original[round_name]
                    if args.max_eval_samples:
                        original_data = original_data.select(range(min(args.max_eval_samples, len(original_data))))

                    for i, example in enumerate(original_data):
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = preds.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(preds.predictions[i].argmax())
                        example_with_prediction['round'] = round_name
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')

        # Calculate and print average accuracy across all rounds
        if anli_results:
            avg_accuracy = sum(r.get('eval_accuracy', 0) for r in anli_results.values()) / len(anli_results)
            print(f'\nAverage ANLI accuracy across all rounds: {avg_accuracy:.4f}')



        #print and save combined confusion matrix for ANLI rounds 
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns       
        all_true_labels = []
        all_predicted_labels = []
        for round_name in ['test_r1', 'test_r2', 'test_r3']:
            if round_name in anli_datasets_featurized:
                eval_dataset = anli_datasets_featurized[round_name]
                eval_predictions = trainer.predict(eval_dataset)
                true_labels = [example['label'] for example in eval_dataset]
                predicted_labels = [int(pred.argmax()) for pred in eval_predictions.predictions]
                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_labels)   
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        print('Combined ANLI Confusion Matrix:')
        print(cm)   
        # Save confusion matrix as image
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                    yticklabels=['Entailment', 'Neutral', 'Contradiction'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Combined ANLI Confusion Matrix')
        plt.savefig(os.path.join(training_args.output_dir, 'confusion_matrix_ANLI.png'))
        plt.close()
        
    

    # Evaluate on contrast sets if requested
    if args.do_eval_contrast:
        print('\nEvaluating on NLI contrast sets...')
        contrast_results = {}
        contrast_predictions = {}

        # Keep reference to original contrast data
        contrast_dataset_original = datasets.load_dataset('AntoineBlanot/snli-contrast')['test']

        for contrast_name in contrast_datasets_featurized:
            print(f'\nEvaluating on {contrast_name}...')
            # Reset eval_predictions for this contrast set
            eval_predictions = None

            # Update trainer's eval dataset
            trainer.eval_dataset = contrast_datasets_featurized[contrast_name]
            contrast_result = trainer.evaluate()
            contrast_results[contrast_name] = contrast_result
            contrast_predictions[contrast_name] = eval_predictions
            print(f'{contrast_name} results:')
            print(contrast_result)

        # Save contrast set results
        if contrast_results:
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, 'contrast_eval_metrics.json'), encoding='utf-8', mode='w') as f:
                json.dump(contrast_results, f, indent=2)

            # Save contrast set predictions
            for contrast_name, preds in contrast_predictions.items():
                if preds is not None:
                    with open(os.path.join(training_args.output_dir, f'{contrast_name}_predictions.jsonl'), encoding='utf-8', mode='w') as f:
                        # Get the processed contrast data with labels
                        contrast_data = contrast_dataset_original

                        # Apply the same label conversion using shared function
                        contrast_data = contrast_data.map(convert_contrast_labels, batched=True)
                        if args.max_eval_samples:
                            contrast_data = contrast_data.select(range(min(args.max_eval_samples, len(contrast_data))))

                        for i, example in enumerate(contrast_data):
                            example_with_prediction = dict(example)
                            example_with_prediction['predicted_scores'] = preds.predictions[i].tolist()
                            example_with_prediction['predicted_label'] = int(preds.predictions[i].argmax())
                            f.write(json.dumps(example_with_prediction))
                            f.write('\n')

            # Print average accuracy if available
            if all('eval_accuracy' in r for r in contrast_results.values()):
                avg_accuracy = sum(r.get('eval_accuracy', 0) for r in contrast_results.values()) / len(contrast_results)
                print(f'\nAverage contrast set accuracy: {avg_accuracy:.4f}')

        # save predictions on contrast sets including premise and hypothesis text
        with open(os.path.join(training_args.output_dir, 'contrast_eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for contrast_name in contrast_datasets_featurized:
                trainer.eval_dataset = contrast_datasets_featurized[contrast_name]
                eval_dataset = contrast_datasets_featurized[contrast_name]
                eval_predictions = trainer.predict(eval_dataset)
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    example_with_prediction['contrast_set'] = contrast_name
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
        # save wrong predictions on contrast sets including premise and hypothesis text
        with open(os.path.join(training_args.output_dir, 'contrast_eval_wrong_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for contrast_name in contrast_datasets_featurized:
                trainer.eval_dataset = contrast_datasets_featurized[contrast_name]
                eval_dataset = contrast_datasets_featurized[contrast_name]
                eval_predictions = trainer.predict(eval_dataset)
                for i, example in enumerate(eval_dataset):
                    predicted_label = int(eval_predictions.predictions[i].argmax())
                    if predicted_label != example['label']:
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = predicted_label
                        example_with_prediction['contrast_set'] = contrast_name
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
        #print and save combined confusion matrix for contrast sets
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns       
        all_true_labels = []
        all_predicted_labels = []
        for contrast_name in contrast_datasets_featurized:  
            eval_dataset = contrast_datasets_featurized[contrast_name]
            eval_predictions = trainer.predict(eval_dataset)
            true_labels = [example['label'] for example in eval_dataset]
            predicted_labels = [int(pred.argmax()) for pred in eval_predictions.predictions]
            all_true_labels.extend(true_labels)
            all_predicted_labels.extend(predicted_labels)   
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        print('Combined Contrast Sets Confusion Matrix:')
        print(cm)   
        # Save confusion matrix as image
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                    yticklabels=['Entailment', 'Neutral', 'Contradiction'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Combined Contrast Sets Confusion Matrix')
        plt.savefig(os.path.join(training_args.output_dir, 'confusion_matrix_Contrast_Sets.png'))
        plt.close()


def print_confusion_matrix_from_file(predictions_file, task='nli'):
    """Print an ASCII confusion matrix from a predictions JSONL file."""

    # Read predictions
    true_labels = []
    predicted_labels = []

    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            true_labels.append(example['label'])
            predicted_labels.append(example['predicted_label'])

    if task == 'nli':
        label_names = ['Entailment', 'Neutral', 'Contradiction']
        label_abbrevs = ['E', 'N', 'C']
    else:
        # For other tasks, use numeric labels
        unique_labels = sorted(set(true_labels + predicted_labels))
        label_names = [f'Class {i}' for i in unique_labels]
        label_abbrevs = [str(i) for i in unique_labels]

    n_classes = len(label_names)

    # Calculate confusion matrix
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1

    # Calculate metrics
    total = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = correct / total if total > 0 else 0

    # Print the matrix
    print("\n" + "="*60)
    print(f"CONFUSION MATRIX - {predictions_file}")
    print("="*60)
    print(f"Total examples: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print()

    # Create ASCII table
    # Header
    col_width = 10
    label_col_width = 20

    # Print header
    print()
    print(" " * (label_col_width + 5) + "PREDICTED")
    print(" " * (label_col_width + 1) + "|" + "".join(f"{abbrev:^{col_width}}" for abbrev in label_abbrevs) + "|")
    print(" " * (label_col_width + 1) + "+" + "-" * (col_width * n_classes) + "+")

    # Rows
    for i, (name, abbrev) in enumerate(zip(label_names, label_abbrevs)):
        row_label = f"{abbrev}: {name[:10]}"

        # Add TRUE label only on middle row for visual clarity
        if i == 1:
            # Middle row gets TRUE label
            full_label = f"TRUE {row_label}"
        else:
            # Other rows get spaces
            full_label = f"     {row_label}"

        print(f"{full_label:<{label_col_width+1}}|", end="")

        for j in range(n_classes):
            count = matrix[i][j]
            # Highlight diagonal (correct predictions)
            if i == j:
                # Use same width as non-diagonal entries but with asterisks
                print(f"  **{count:4}**", end="")
            else:
                print(f"    {count:4}  ", end="")

        # Print row metrics
        row_total = matrix[i].sum()
        row_recall = matrix[i][i] / row_total if row_total > 0 else 0
        print(f"|  {row_total:4} ({row_recall:5.1%})")

    print(" " * (label_col_width + 1) + "+" + "-" * (col_width * n_classes) + "+")

    # Column totals
    print(" " * (label_col_width + 1) + "|", end="")
    for j in range(n_classes):
        col_total = matrix[:, j].sum()
        print(f"  {col_total:5}  ", end="")
    print("|")

    # Precision row
    print(" " * (label_col_width - 4) + "Prec:|", end="")
    for j in range(n_classes):
        col_total = matrix[:, j].sum()
        precision = matrix[j][j] / col_total if col_total > 0 else 0
        print(f" {precision:6.1%} ", end="")
    print("|")

    # Per-class F1 scores
    print("\nPer-class F1 scores:")
    for i, name in enumerate(label_names):
        row_total = matrix[i].sum()
        col_total = matrix[:, i].sum()
        recall = matrix[i][i] / row_total if row_total > 0 else 0
        precision = matrix[i][i] / col_total if col_total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name:15}: {f1:.4f}")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Check if we're just printing a confusion matrix
    import sys
    if len(sys.argv) >= 3 and '--print_confusion_matrix' in sys.argv:
        idx = sys.argv.index('--print_confusion_matrix')
        if idx + 1 < len(sys.argv):
            predictions_file = sys.argv[idx + 1]
            if os.path.exists(predictions_file):
                # Determine task type from filename or default to nli
                task = 'nli'  # Default to NLI for now
                print_confusion_matrix_from_file(predictions_file, task)
                sys.exit(0)
            else:
                print(f"Error: Predictions file '{predictions_file}' not found.")
                sys.exit(1)

    main()
