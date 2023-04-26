"""
This file is modified from the original code from: https://github.com/princeton-nlp/LM-BFF/src/processors.py
Dataset utils for different data settings.
"""

import os
import copy
import logging
import numpy as np
import json
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from copy import deepcopy
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Semi-supervised benchmarks
class ReverseSst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )   

    def get_train_examples(self, file_path, seed=None, num_labelled_data=None):
        """See base class."""
        return self._create_examples(self.read_json_file(os.path.join(file_path, "train.json")), "train", seed=seed, num_labelled_data=num_labelled_data, file_path=file_path)

    def get_dev_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "dev.json")
        return self._create_examples(self.read_json_file(file_path), "dev")

    def get_test_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "test.json")
        return self._create_examples(self.read_json_file(file_path), "test")

    def get_psuedo_examples(self, file_path):
        """See base class."""
        return self._create_examples(self.read_json_file(file_path), "pseudo")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_to_word_mapping(self):
        """Maps the label to the corresponding token."""
        return {"0": "great", "1": "terrible"}

    def _create_examples(self, lines, set_type, seed=None, num_labelled_data=None, file_path=None):
        """Creates examples for the training, dev and test sets."""
        if set_type == "train" and num_labelled_data is not None:
            lb_idx = np.load(os.path.join(file_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labelled_data, seed)))
        examples = []
        for idx, (sample_id, key) in enumerate(lines.items()):
            if set_type == "train" and num_labelled_data is not None and idx not in lb_idx:
                continue
            guid = "%s-%s" % (set_type, int(sample_id))
            text_a = key["ori"]
            label = key["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def read_json_file(self, file_path):
        """Reads a json file."""
        with open(file_path, "r") as f:
            return json.load(f)
   

class AmazonReviewProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, file_path, seed=None, num_labelled_data=None):
        """See base class."""
        return self._create_examples(self.read_json_file(os.path.join(file_path, "train.json")), "train", seed=seed, num_labelled_data=num_labelled_data, file_path=file_path)

    def get_dev_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "dev.json")
        return self._create_examples(self.read_json_file(file_path), "dev")

    def get_test_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "test.json")
        return self._create_examples(self.read_json_file(file_path), "test")

    def get_psuedo_examples(self, file_path):
        """See base class."""
        return self._create_examples(self.read_json_file(file_path), "pseudo")
    
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def get_label_to_word_mapping(self):
        """Maps the label to the corresponding token."""
        return {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}

    def _create_examples(self, lines, set_type, seed=None, num_labelled_data=None, file_path=None):
        """Creates examples for the training, dev and test sets."""
        if set_type == "train" and num_labelled_data is not None:
            lb_idx = np.load(os.path.join(file_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labelled_data, seed)))
        examples = []
        for idx, (sample_id, key) in enumerate(lines.items()):
            if set_type == "train" and num_labelled_data is not None and idx not in lb_idx:
                continue
            guid = "%s-%s" % (set_type, int(sample_id))
            text_a = key["ori"]
            label = key["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def read_json_file(self, file_path):
        """Reads a json file."""
        with open(file_path, "r") as f:
            return json.load(f)


class YahooAnswerProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )  

    def get_train_examples(self, file_path, seed=None, num_labelled_data=None):
        """See base class."""
        return self._create_examples(self.read_json_file(os.path.join(file_path, "train.json")), "train", seed=seed, num_labelled_data=num_labelled_data, file_path=file_path)

    def get_dev_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "dev.json")
        return self._create_examples(self.read_json_file(file_path), "dev")

    def get_test_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "test.json")
        return self._create_examples(self.read_json_file(file_path), "test")

    def get_psuedo_examples(self, file_path):
        """See base class."""
        return self._create_examples(self.read_json_file(file_path), "pseudo")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def get_label_to_word_mapping(self):
        """Maps the label to the corresponding token."""
        return {"0": "culture", "1": "science", "2": "health", "3": "education", "4": "computer", "5": "sports", "6": "business", "7": "music", "8": "family", "9": "politics"}

    def _create_examples(self, lines, set_type, seed=None, num_labelled_data=None, file_path=None):
        """Creates examples for the training, dev and test sets."""
        if set_type == "train" and num_labelled_data is not None:
            lb_idx = np.load(os.path.join(file_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labelled_data, seed)))
        examples = []
        for idx, (sample_id, key) in enumerate(lines.items()):
            if set_type == "train" and num_labelled_data is not None and idx not in lb_idx:
                continue
            guid = "%s-%s" % (set_type, int(sample_id))
            text_a = key["ori"]
            label = key["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def read_json_file(self, file_path):
        """Reads a json file."""
        with open(file_path, "r") as f:
            return json.load(f)


class YelpReviewProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        ) 

    def get_train_examples(self, file_path, seed=None, num_labelled_data=None):
        """See base class."""
        return self._create_examples(self.read_json_file(os.path.join(file_path, "train.json")), "train", seed=seed, num_labelled_data=num_labelled_data, file_path=file_path)

    def get_dev_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "dev.json")
        return self._create_examples(self.read_json_file(file_path), "dev")

    def get_test_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "test.json")
        return self._create_examples(self.read_json_file(file_path), "test")

    def get_psuedo_examples(self, file_path):
        """See base class."""
        return self._create_examples(self.read_json_file(file_path), "pseudo")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def get_label_to_word_mapping(self):
        """Maps the label to the corresponding token."""
        return {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}

    def _create_examples(self, lines, set_type, seed=None, num_labelled_data=None, file_path=None):
        """Creates examples for the training, dev and test sets."""
        if set_type == "train" and num_labelled_data is not None:
            lb_idx = np.load(os.path.join(file_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labelled_data, seed)))
        examples = []
        for idx, (sample_id, key) in enumerate(lines.items()):
            if set_type == "train" and num_labelled_data is not None and idx not in lb_idx:
                continue
            guid = "%s-%s" % (set_type, int(sample_id))
            text_a = key["ori"]
            label = key["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def read_json_file(self, file_path):
        """Reads a json file."""
        with open(file_path, "r") as f:
            return json.load(f)


class AGNewsProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        ) 

    def get_train_examples(self, file_path, seed=None, num_labelled_data=None):
        """See base class."""
        return self._create_examples(self.read_json_file(os.path.join(file_path, "train.json")), "train", seed=seed, num_labelled_data=num_labelled_data, file_path=file_path)

    def get_dev_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "dev.json")
        return self._create_examples(self.read_json_file(file_path), "dev")

    def get_test_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "test.json")
        return self._create_examples(self.read_json_file(file_path), "test")

    def get_psuedo_examples(self, file_path):
        """See base class."""
        return self._create_examples(self.read_json_file(file_path), "pseudo")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_label_to_word_mapping(self):
        """Maps the label to the corresponding token."""
        return {"0": "world", "1": "sports", "2": "business", "3": "tech"}

    def _create_examples(self, lines, set_type, seed=None, num_labelled_data=None, file_path=None):
        """Creates examples for the training, dev and test sets."""
        if set_type == "train" and num_labelled_data is not None:
            lb_idx = np.load(os.path.join(file_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labelled_data, seed)))
        examples = []
        for idx, (sample_id, key) in enumerate(lines.items()):
            if set_type == "train" and num_labelled_data is not None and idx not in lb_idx:
                continue
            guid = "%s-%s" % (set_type, int(sample_id))
            text_a = key["ori"]
            label = key["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def read_json_file(self, file_path):
        """Reads a json file."""
        with open(file_path, "r") as f:
            return json.load(f)


class IMDBProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )   

    def get_train_examples(self, file_path, seed=None, num_labelled_data=None):
        """See base class."""
        return self._create_examples(self.read_json_file(os.path.join(file_path, "train.json")), "train", seed=seed, num_labelled_data=num_labelled_data, file_path=file_path)

    def get_dev_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "dev.json")
        return self._create_examples(self.read_json_file(file_path), "dev")

    def get_test_examples(self, file_path):
        """See base class."""
        file_path = os.path.join(file_path, "test.json")
        return self._create_examples(self.read_json_file(file_path), "test")

    def get_psuedo_examples(self, file_path):
        """See base class."""
        return self._create_examples(self.read_json_file(file_path), "pseudo")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_to_word_mapping(self):
        """Maps the label to the corresponding token."""
        return {"0": "great", "1": "terrible"}

    def _create_examples(self, lines, set_type, seed=None, num_labelled_data=None, file_path=None):
        """Creates examples for the training, dev and test sets."""
        if set_type == "train" and num_labelled_data is not None:
            lb_idx = np.load(os.path.join(file_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labelled_data, seed)))
        examples = []
        for idx, (sample_id, key) in enumerate(lines.items()):
            if set_type == "train" and num_labelled_data is not None and idx not in lb_idx:
                continue
            guid = "%s-%s" % (set_type, int(sample_id))
            text_a = key["ori"]
            label = key["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def read_json_file(self, file_path):
        """Reads a json file."""
        with open(file_path, "r") as f:
            return json.load(f)


# Few Shot learning benchmarks
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        text_index = 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = float(v["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            # label = '0' if float(label) <= 2.5 else '1'
            label = float(label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 3
        q2_index = 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def get_psuedo_examples(self, file_path):
        """Gets a collection of [`InputExample`] for the psuedo set."""
        with open(file_path) as f:
            lines = json.load(f)
        examples = []
        for i, (k, v) in enumerate(lines.items()):
            guid = "%s-%s" % ("pseudo", i)
            text_a = v["text_a"]
            text_b = v["text_b"]
            label = v["label"]
            if text_a != text_a:
                continue # During the fully-supervised learning, some of examples in mpqa are Null.
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def __init__(self, task_name):
        self.task_name = task_name 

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
  
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        else:
            raise Exception("task_name not supported.")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(InputExample(guid=guid, text_a=line[1] + '. ' + line[2], short_text=line[1] + ".", label=line[0]))
            elif self.task_name == "yelp_review_full":
                examples.append(InputExample(guid=guid, text_a=line[1], short_text=line[1], label=line[0]))
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += ' ' + line[2]
                if not pd.isna(line[3]):
                    text += ' ' + line[3]
                examples.append(InputExample(guid=guid, text_a=text, short_text=line[1], label=line[0])) 
            elif self.task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
                if self.task_name == 'mpqa' and line[1] != line[1]:
                    continue # During the fully-supervised learning, some of examples in mpqa are Null.
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples


def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

# Templates for one sentence text classification tasks
def sst2_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "It was", tokenizer.mask_token, "."], []

def sst5_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "It was", tokenizer.mask_token, "."], []

def mr_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "It was", tokenizer.mask_token, "."], []

def cr_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "It was", tokenizer.mask_token, "."], []

def mpqa_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "is", tokenizer.mask_token, "."], []

def subj_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "This is", tokenizer.mask_token, "."], []

def trec_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [tokenizer.mask_token, ":", text_a, text_b], []

def cola_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "This is", tokenizer.mask_token], []

# Templates for two sentence text classification tasks
def mrpc_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, tokenizer.mask_token, ",", text_b], []

def qqp_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, tokenizer.mask_token, ",", text_b], []

def sstb_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, tokenizer.mask_token, ",", text_b], []

def mnli_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, "?", tokenizer.mask_token, ",", text_b], []

def snli_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, ".", tokenizer.mask_token, ", in this case", text_b], []

def qnli_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, "?", tokenizer.mask_token, ",", text_b], []

def rte_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, ".", tokenizer.mask_token, ", I think that", text_b], []

# def dart_template(text_a: str, text_b: str, label: str):
#     return [text_a, text_b, "It was", label]

# def dart_ts_template(text_a: str, text_b: str, label: str):
#     return [text_a, label, ",", text_b]

# Templates for semi-supervised tasks
def ssl_template(text_a: str, text_b: str, tokenizer: PreTrainedTokenizer):
    return [text_a, text_b, "It was", tokenizer.mask_token, "."], []


template_mapping = {
    # One sentence tasks
    "SST-2": sst2_template,
    "sst-5": sst5_template,
    "mr": mr_template,
    "cr": cr_template,
    "mpqa": mpqa_template,
    "subj": subj_template,
    "trec": trec_template,
    "CoLA": cola_template,
    
    # Two sentence tasks
    'MRPC': mrpc_template,
    'QQP': qqp_template,
    'STS-B': sstb_template,
    'MNLI': mnli_template,
    'MNLI-mm': mnli_template,
    'SNLI': snli_template,
    'QNLI': qnli_template,
    'RTE': rte_template,

    # Semi-supervised tasks
    "reverse_SST-2": sst2_template,
    "aclImdb": ssl_template,
    "ag_news": ssl_template,
    "yelp_review": ssl_template,
    "yahoo_answers": ssl_template,
    "amazon_review": ssl_template,
}

map_of_mapping = {
    # One sentence tasks
    'SST-2': {'0':'terrible','1':'great'},
    'sst-5': {0:'terrible', 1:'bad', 2:'okay', 3:'good', 4:'great'},
    'mr': {0:'terrible', 1:'great'},
    'cr': {0:'terrible', 1:'great'},
    'subj': {0:'subjective', 1:'objective'},
    'trec': {0:'Description', 1:'Entity', 2:'Expression', 3:'Human', 4:'Location', 5:'Number'},
    'mpqa': {0:'negative', 1:'positive'},
    'CoLA': {'0':'incorrect', '1':'correct'},

    # Two sentence tasks
    'MRPC': {'0':'No', '1':'Yes'},
    'QQP': {'0':'No', '1':'Yes'},
    'STS-B': {'0':'No', '1':'Yes'},
    'MNLI': {'contradiction':'No', 'entailment':'Yes', 'neutral':'Maybe'},
    'MNLI-mm': {'contradiction':'No', 'entailment':'Yes', 'neutral':'Maybe'},
    'SNLI': {'contradiction':'No', 'entailment':'Yes', 'neutral':'Maybe'},
    'QNLI': {'not_entailment':'No', 'entailment':'Yes'},
    # 'RTE': {'not_entailment':'No', 'entailment':'Yes'}
    'RTE': {'not_entailment':'Yet', 'entailment':'Clearly'},

    "aclImdb": {"0": "great", "1": "terrible"},
    "reverse_SST-2": {"0": "great", "1": "terrible"},
    "ag_news": {"0": "world", "1": "sports", "2": "business", "3": "tech"},
    "yelp_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    "amazon_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    "yahoo_answers": {"0": "culture", "1": "science", "2": "health", "3": "education", "4": "computer", "5": "sports", "6": "business", "7": "music", "8": "family", "9": "politics"},
}

processors_mapping = {
    "cola": ColaProcessor(),
    "mnli": MnliProcessor(),
    "mnli-mm": MnliMismatchedProcessor(),
    "mrpc": MrpcProcessor(),
    "sst-2": Sst2Processor(),
    "sts-b": StsbProcessor(),
    "qqp": QqpProcessor(),
    "qnli": QnliProcessor(),
    "rte": RteProcessor(),
    "wnli": WnliProcessor(),
    "snli": SnliProcessor(),
    "mr": TextClassificationProcessor("mr"),
    "sst-5": TextClassificationProcessor("sst-5"),
    "subj": TextClassificationProcessor("subj"),
    "trec": TextClassificationProcessor("trec"),
    "cr": TextClassificationProcessor("cr"),
    "mpqa": TextClassificationProcessor("mpqa"),
    
    # Semi-supervised tasks
    "aclimdb": IMDBProcessor(),
    "ag_news": AGNewsProcessor(),
    "yelp_review": YelpReviewProcessor(),
    "yahoo_answers": YahooAnswerProcessor(),
    "amazon_review": AmazonReviewProcessor(),
    "reverse_sst-2": ReverseSst2Processor(),
}

num_labels_mapping = {
    # Few-shot tasks
    "cola": 2,
    "mnli": 3,
    "mnli-mm": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli": 3,
    "mr": 2,
    "sst-5": 5,
    "subj": 2,
    "trec": 6,
    "cr": 2,
    "mpqa": 2,

    # Semi-supervised tasks
    "reverse_sst-2": 2,
    "aclimdb": 2,
    "ag_news": 4,
    "yelp_review": 5,
    "amazon_review": 5,
    "yahoo_answers": 10,
}

output_modes_mapping = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "snli": "classification",
    "mr": "classification",
    "sst-5": "classification",
    "subj": "classification",
    "trec": "classification",
    "cr": "classification",
    "mpqa": "classification",

    "aclimdb": "classification",
    "reverse_sst-2": "classification",
    "ag_news": "classification",
    "yelp_review": "classification",
    "amazon_review": "classification",
    "yahoo_answers": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "cola": glue_compute_metrics,
    "mnli": glue_compute_metrics,
    "mnli-mm": glue_compute_metrics,
    "mrpc": glue_compute_metrics,
    "sst-2": glue_compute_metrics,
    "sts-b": glue_compute_metrics,
    "qqp": glue_compute_metrics,
    "qnli": glue_compute_metrics,
    "rte": glue_compute_metrics,
    "wnli": glue_compute_metrics,
    "snli": text_classification_metrics,
    "mr": text_classification_metrics,
    "sst-5": text_classification_metrics,
    "subj": text_classification_metrics,
    "trec": text_classification_metrics,
    "cr": text_classification_metrics,
    "mpqa": text_classification_metrics,
}


evaluate_metrics_mapping = {
    # One sentence tasks
    "SST-2": "eval_acc",
    "sst-5": "eval_acc",
    "mr": "eval_acc",
    "cr": "eval_acc",
    "subj": "eval_acc",
    "trec": "eval_acc",
    "mpqa": "eval_acc",
    "CoLA": "eval_mcc",
    # Two sentence tasks
    "MRPC": "eval_f1",
    "QQP": "eval_f1",
    "STS-B": "eval_pearson",
    "MNLI": "eval_mnli/acc",
    "SNLI": "eval_acc",
    "QNLI": "eval_acc",
    "RTE": "eval_acc",
}
