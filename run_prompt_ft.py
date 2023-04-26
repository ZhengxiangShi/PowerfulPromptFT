""" Finetuning the models for sequence classification on downstream tasks."""

import os
import json
import sys
import copy
import tqdm
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

import torch
from src.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping, evaluate_metrics_mapping
from src.model import RobertaForPromptingClassification
from src.dataset import SemiLMDatasetSequenceClassification, DartDatasetSequenceClassification
from transformers import InputExample

import evaluate
import transformers
from transformers import (
    RobertaTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_embedding_first: bool = field(default=False, metadata={"help": "Whether train the embeddings of the model first."})
    downstream_task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on"},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on."},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    truncate_head: bool = field(
        default=False, metadata={"help": "Truncate the head or tail of the sequence."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_labelled_data: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    task_type: str = field(default="glue", metadata={"help": "The type of the task."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default="prompting",
        metadata={"help": "Select prompting, dart, or mask for the model type"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    eb_learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW for embedding training."})
    eb_num_train_epochs: float = field(default=5.0, metadata={"help": "Total number of training epochs to perform for embedding training."})
    run_pseduo_label: bool = field(default=False, metadata={"help": "Whether to run pseudo label."})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logger.info(f"Runing task_type: {data_args.task_type}")
    if data_args.task_type == "glue":
        training_args.metric_for_best_model = evaluate_metrics_mapping[data_args.downstream_task_name]
        training_args.greater_is_better = True
        logger.info("metric_for_best_model is set to {}".format(training_args.metric_for_best_model))
    else:
        training_args.metric_for_best_model = "eval_f1"
        training_args.greater_is_better = True
        logger.info("metric_for_best_model is set to {}".format(training_args.metric_for_best_model))

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(training_args.output_dir, 'output.log'), mode='w'),
                  logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_args.seed = training_args.seed

    try:
        num_labels = num_labels_mapping[data_args.downstream_task_name.lower()]
        output_mode = output_modes_mapping[data_args.downstream_task_name.lower()]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.downstream_task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.downstream_task_name))

    # Loading a dataset from your local files.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    logger.info(f'before {len(tokenizer)}')

    # Preprocessing the raw_datasets
    if model_args.model_type == "prompting":
        ModelSpecificDataset = SemiLMDatasetSequenceClassification
    elif model_args.model_type == "dart":
        ModelSpecificDataset = DartDatasetSequenceClassification
    else:
        raise NotImplementedError(f"model type {model_args.model_type} is not implemented")

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if training_args.do_train:
        train_dataset = ModelSpecificDataset(data_args, tokenizer=tokenizer, file_path=data_files["train"], mode="train")
    if training_args.do_eval:
        eval_dataset = ModelSpecificDataset(data_args, tokenizer=tokenizer, file_path=data_files["validation"], mode="dev")
    if training_args.do_predict:
        if model_args.run_pseduo_label:
            # We use train and dev set with pseduo label for mlm training
            predict_dataset_train = ModelSpecificDataset(data_args, tokenizer=tokenizer, file_path=data_files["test"], mode="pseudo_train")
            predict_dataset_dev = ModelSpecificDataset(data_args, tokenizer=tokenizer, file_path=data_files["test"], mode="pseudo_dev")
        else:
            # Otherwise, we use test set for reuglar evaluation
            predict_dataset = ModelSpecificDataset(data_args, tokenizer=tokenizer, file_path=data_files["test"], mode="test")
    processed_dataset = train_dataset if train_dataset is not None else eval_dataset if eval_dataset is not None else predict_dataset
    logger.info(f'after {len(tokenizer)}')
    if model_args.model_type == "dart":
        logger.info(f'pattern: {processed_dataset.pattern}')

    # Load pretrained model
    if model_args.model_type == "prompting" or model_args.model_type == "dart":
        model = RobertaForPromptingClassification.from_pretrained(model_args.model_name_or_path)
    else:
        raise NotImplementedError(f"model type {model_args.model_type} is not implemented")

    model.mlm_logits_to_cls_logits_tensor = processed_dataset.mlm_logits_to_cls_logits_tensor
    model.num_labels = 1 if data_args.downstream_task_name == "STS-B" else len(processed_dataset.word_to_label)
    if data_args.downstream_task_name == "STS-B":
        model.lb = 0
        model.ub = 5
    logger.info("word_to_label: {}".format(processed_dataset.word_to_label))
    if (model_args.model_type == "dart" or model_args.model_type == "mask") and training_args.do_train:
        """
        Only for DART model, we need to resize the token embeddings and initialize the new embeddings from the token embeddings. 
        Because Dart model uses the additional token embeddings for the pattern and the label.
        We will not update embeddings when the model is evaluated or predicted, with the assumption that the model is already trained.
        """
        model.resize_token_embeddings(len(tokenizer))
        model._init_embedding(processed_dataset.pattern_index, processed_dataset.mlm_label_index, initialize_from_token_embeddings=True)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set guid: {train_dataset.examples[index].guid}.")
            logger.info(f"Sample {index} of the training set text_a: {train_dataset.examples[index].text_a}.")
            logger.info(f"Sample {index} of the training set text_b: {train_dataset.examples[index].text_b}.")
            logger.info(f"Sample {index} of the training set label: {train_dataset.examples[index].label}.")
            logger.info(f"Sample {index} of the training set ids: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_type == "ssl" :
        f1_metric = evaluate.load("f1")
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = f1_metric.compute(predictions=preds, references=p.label_ids, average='macro')
            return result
    elif data_args.task_type == "glue":
        def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                # Note: the eval dataloader is sequential, so the examples are in order.
                # We average the logits over each sample for using demonstrations.
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                num_logits = preds.shape[-1]
                if num_logits == 1:
                    preds = np.squeeze(preds)
                else:
                    preds = np.argmax(preds, axis=1)
                return compute_metrics_mapping[task_name](task_name, preds, p.label_ids)
            return compute_metrics_fn
    else:
        raise NotImplementedError(f"task type {model_args.task_type} is not implemented")
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if data_args.task_type == "ssl" else build_compute_metrics_fn(data_args.downstream_task_name.lower()),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if model_args.model_type == "mask":
            logger.info("mask_map: {}".format(model.mask_map))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if not model_args.run_pseduo_label:
            test_output = trainer.predict(predict_dataset)
            test_metrics = test_output.metrics
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            if data_args.downstream_task_name == "MNLI":
                data_args.downstream_task_name = "MNLI-mm"
                predict_dataset_2 = ModelSpecificDataset(data_args, tokenizer=tokenizer, file_path=data_files["test"], mode="test")
                test_output = trainer.predict(predict_dataset_2)
                test_metrics = test_output.metrics
                trainer.log_metrics("test_mm", test_metrics)
                trainer.save_metrics("test_mm", test_metrics)
        else:
            for predict_dataset, split in zip([predict_dataset_train, predict_dataset_dev], ["train", "dev"]):
                test_output = trainer.predict(predict_dataset)
                test_metrics = test_output.metrics
                trainer.log_metrics("test", test_metrics)
                trainer.save_metrics("test", test_metrics)
                num_logits = test_output.predictions.shape[-1]
                if num_logits == 1:
                    label_ids = test_output.label_ids.tolist()
                    test_probs = np.squeeze(test_output.predictions).tolist()
                    test_predictions = ['0' if float(label) <= 2.5 else '1' for label in test_probs]
                else:
                    test_logits = torch.tensor(test_output.predictions)
                    test_logits = torch.nn.functional.softmax(test_logits.float(), dim=1)
                    test_probs, test_predictions = torch.topk(test_logits, 1)
                    test_probs, test_predictions = test_probs.squeeze(1).tolist(), test_predictions.squeeze(1).tolist()
                    label_ids = test_output.label_ids.tolist() if test_output.label_ids is not None else [None] * len(test_predictions)

                output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{split}.json")
                output_dct = {}
                test_pred_list = []
                test_gold_list = []
                with open(output_predict_file, "w") as f_writer:
                    logger.info(f"***** Predict results {data_args.downstream_task_name} *****")
                    for index, (_label, _pred, _confidence) in enumerate(tqdm.tqdm(zip(label_ids, test_predictions, test_probs))):
                        test_sample = predict_dataset.__getitem__(index)  # Dict
                        test_InputExample = predict_dataset.examples[index]  # InputExample
                        assert isinstance(test_InputExample, InputExample)
                        assert predict_dataset.convert_fn(test_InputExample).input_ids == test_sample["input_ids"]
                        # assert _label == test_sample["labels"]
                        if data_args.downstream_task_name == "STS-B":
                            text_label = "0" if float(test_InputExample.label) <= 2.5 else "1"
                            text_label = processed_dataset.label_to_word[text_label]
                        else:
                            text_label = processed_dataset.label_to_word[test_InputExample.label] if test_InputExample.label else None
                        output_dct[str(index)] = {
                            "text_a": test_InputExample.text_a,
                            "text_b": test_InputExample.text_b,
                            "text_label": text_label,
                            "label": str(_label),
                            "pred": str(_pred),
                            "confidence": _confidence,
                        }
                        if data_args.downstream_task_name == "STS-B":
                            test_pred_list.append(_confidence)
                        else:
                            test_pred_list.append(_pred)
                        test_gold_list.append(_label)
                    if data_args.task_type == "ssl":
                        test_result = f1_metric.compute(predictions=test_pred_list, references=test_gold_list, average='macro')
                    else:
                        if test_InputExample.label is not None:
                            test_result = compute_metrics_mapping[data_args.downstream_task_name.lower()](data_args.downstream_task_name.lower(), np.array(test_pred_list), np.array(test_gold_list))
                        else:
                            test_result = {}
                    logger.info(f"***** {data_args.downstream_task_name} test results: {test_result} *****")
                    f_writer.write(json.dumps(output_dct, indent=4))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.downstream_task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = data_args.task_type
        kwargs["dataset_args"] = data_args.downstream_task_name
        kwargs["dataset"] = data_args.downstream_task_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()