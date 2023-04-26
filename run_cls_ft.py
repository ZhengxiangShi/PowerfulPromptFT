""" Finetuning the models for sequence classification on downstream tasks."""

import os
import sys
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

import datasets
from src.dataset import CLSDataset
from src.model import RobertaForSequenceClassification
from src.processors import processors_mapping, compute_metrics_mapping, evaluate_metrics_mapping

import transformers
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import RobertaTokenizer, RobertaConfig, InputExample


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on."}
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
        default=True, metadata={"help": "Truncate the head or tail of the sequence."}
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    task_type: str = field(default=None, metadata={"help": "The type of the task."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    logger.info(f"Runing task_type: {data_args.task_type}")
    if data_args.task_type == "glue":
        training_args.metric_for_best_model = evaluate_metrics_mapping[data_args.task_name]
        training_args.greater_is_better = True
        logger.info("metric_for_best_model is set to {}".format(training_args.metric_for_best_model))
    elif data_args.task_type == "ssl":
        training_args.metric_for_best_model = "eval_f1"
        training_args.greater_is_better = True
        logger.info("metric_for_best_model is set to {}".format(training_args.metric_for_best_model))
    else:
        raise ValueError("task_type should be either glue or ssl")

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
    datasets.utils.logging.set_verbosity(log_level)
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

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {}
    if training_args.do_train:
        data_files["train"] = data_args.train_file
    if training_args.do_eval:
        data_files["validation"] = data_args.validation_file
    if training_args.do_predict:
        data_files["test"] = data_args.test_file

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    processor = processors_mapping[data_args.task_name.lower()]
    label_to_id = {v: i for i, v in enumerate(processor.get_labels())}
    if data_args.task_name == "STS-B": label_to_id = {"0": 0, "1": 1}
    num_labels = len(label_to_id)
    # Load pretrained model and tokenizer
    config = RobertaConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        add_prefix_space=True,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        if data_args.task_name.lower() == "mpqa":
            # During the fully-supervised learning, some of examples in mpqa are Null.
            args = ([e.text_a for e in examples if e.text_a == e.text_a], )
        else:
            args = (
                ([e.text_a for e in examples],) if examples[0].text_b is None else ([e.text_a for e in examples], [e.text_b for e in examples])
            )
        result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
        if data_args.task_name.lower() == "sts-b":
            result["label"] = [float(e.label) for e in examples]
        else:
            result["label"] = [(label_to_id[e.label] if e.label != -1 else -1) for e in examples]
        return result

    if training_args.do_train:
        train_examples = processor.get_train_examples(data_files["train"])
        train_dataset = CLSDataset(preprocess_function(train_examples))
    if training_args.do_eval:
        eval_examples = processor.get_dev_examples(data_files["validation"])
        eval_dataset = CLSDataset(preprocess_function(eval_examples))
    if training_args.do_predict:
        test_examples = processor.get_test_examples(data_files["test"])
        predict_dataset = CLSDataset(preprocess_function(test_examples))

    model = RobertaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    if data_args.task_name == "STS-B":
        model.lb = 0
        model.ub = 5
        model.config.problem_type = "regression"

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=build_compute_metrics_fn(data_args.task_name.lower()),
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

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        test_output = trainer.predict(predict_dataset)
        test_metrics = test_output.metrics
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        if data_args.task_name == "MNLI":
            processor = processors_mapping["mnli-mm"]
            test_examples = processor.get_test_examples(data_files["test"])
            predict_dataset_2 = CLSDataset(preprocess_function(test_examples))
            test_output = trainer.predict(predict_dataset_2)
            test_metrics = test_output.metrics
            trainer.log_metrics("test_mm", test_metrics)
            trainer.save_metrics("test_mm", test_metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = data_args.task_type
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = data_args.task_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()