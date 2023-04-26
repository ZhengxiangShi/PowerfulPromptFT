"""Dataset utils for different data settings for 21 different datasets."""

import os
import json
import logging
import torch
import time
import tqdm
import dataclasses
from dataclasses import dataclass
from src.processors import processors_mapping, map_of_mapping, template_mapping
from transformers.data.processors.utils import InputFeatures
from transformers import GPT2Tokenizer, RobertaTokenizer, InputExample
from typing import Tuple, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mlms: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

      
class SemiLMDatasetSequenceClassification(torch.utils.data.Dataset):
    """Few-shot dataset."""
    def __init__(self, args, tokenizer, file_path, mode="train"):
        self.args = args
        self.task_name = args.downstream_task_name
        self.processor = processors_mapping[self.task_name.lower()]
        self.tokenizer = tokenizer
        self.kwargs = {'add_prefix_space': True} if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, RobertaTokenizer) else {}
        self.mode = mode
        self.file_path = file_path

        # assert mode in ["train", "dev", "test"]

        self.max_length = args.max_seq_length

        # Get label list and its mapping to word
        self.label_to_word = map_of_mapping[self.task_name]
        self.word_to_label = {v: k for k, v in self.label_to_word.items()}
        self.label_map = {label: i for i, label in enumerate(self.label_to_word.keys())}
        self.build_mlm_logits_to_cls_logits_tensor(self.get_label_words())

        # Load cache
        cache_path, file_name = os.path.split(self.file_path)
        file_name = file_name.split(".")[0]

        logger.info(f"Creating examples from dataset file at {self.file_path}")
        
        start = time.time()

        if self.mode == "train" and self.task_name in ["aclImdb", "ag_news", "yelp_review", "yahoo_answers", "amazon_review", "reverse_SST-2"]:
            self.examples = self.processor.get_train_examples(self.file_path, seed=self.args.seed, num_labelled_data=self.args.num_labelled_data)
        elif self.mode == "train":
            self.examples = self.processor.get_train_examples(self.file_path)
        elif self.mode == "dev":
            self.examples = self.processor.get_dev_examples(self.file_path)
        elif self.mode == "test":
            self.examples = self.processor.get_test_examples(self.file_path)
        elif "pseudo" in self.mode:
            if "train" in self.mode:
                self.examples = self.processor.get_psuedo_examples(os.path.join(self.file_path, "train.json"))
            elif "dev" in self.mode:
                self.examples = self.processor.get_psuedo_examples(os.path.join(self.file_path, "dev.json"))
        else:
            raise ValueError("Invalid mode: %s" % self.mode)

        logger.info(f"Creating {self.mode} features from dataset file at {self.file_path}")

        self.features = []

        for ex in tqdm.tqdm(self.examples, desc="Creating {} features".format(self.mode)):
                self.features.append(self.convert_fn(ex))

        self.num_sample = len(self.features)
        logger.info("Getting {} {} samples in total.".format(self.num_sample, self.mode))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return {"input_ids": self.features[i].input_ids,
                "attention_mask": self.features[i].attention_mask,
                "labels": self.features[i].label,
                "mlms": self.features[i].mlms}

    def enc(self, label):
        return self.tokenizer.encode(label, add_special_tokens=False, **self.kwargs)
    
    def get_label_words(self):
        return list(self.label_to_word.values())

    def convert_fn(self, example):
        """
        Returns a list of processed "InputFeatures".
        """
        input_ids, token_type_ids = self.encode(example)
        if self.task_name.lower() == "sts-b":
            # STS-B is a regression task
            label = float(example.label)
        else:
            label = self.label_map[example.label] if example.label is not None else None

        mlm_label = [-100] * len(input_ids)
        if self.task_name.lower() == "sts-b":
            # STS-B is a regression task
            hard_label = '0' if float(example.label) <= 2.5 else '1'
            new_label = self.enc(self.label_to_word[hard_label])
        else:
            new_label = self.enc(self.label_to_word[example.label]) if example.label is not None else [1]
        assert len(new_label) == 1
        mlm_index = input_ids.index(self.tokenizer.mask_token_id)
        mlm_label[mlm_index] = new_label[0]
        attention_mask = [1] * len(input_ids)

        # Pad
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            mlm_label.append(-100)

        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        assert len(mlm_label) == self.max_length
        assert sum([1 if i > 0 else 0 for i in mlm_label])
        
        return OurInputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label,
            mlms=mlm_label,
        )

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(self.tokenizer.encode(x, add_special_tokens=False, **self.kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(self.tokenizer.encode(x, add_special_tokens=False, **self.kwargs), s) for x, s in parts_b if x]

        num_special = self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(parts_a, parts_b, max_length=self.max_length - num_special)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else []

        if tokens_b:
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        else:
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a)

        return input_ids, token_type_ids
    
    def get_parts(self, example: InputExample):
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        return template_mapping[self.task_name](text_a, text_b, self.tokenizer)
        # return [text_a, text_b, "In summary, the movie is", self.tokenizer.mask_token], []
        # return [self.tokenizer.mask_token, ':', text_a, text_b], []
        
    def shortenable(self, s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True
    
    def _seq_length(self, parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    def _remove_last(self, parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    def build_mlm_logits_to_cls_logits_tensor(self, labels: List[str]):
        self.mlm_logits_to_cls_logits_tensor = torch.zeros(len(labels), dtype=torch.long, requires_grad=False)
        for i, label in enumerate(labels):
            self.mlm_logits_to_cls_logits_tensor[i] = self.get_verbalization_ids(label, force_single_token=True)
    
    def get_verbalization_ids(self, word: str, force_single_token: bool) -> Union[int, List[int]]:
        """
        Get the token ids corresponding to a verbalization

        :param word: the verbalization
        :param tokenizer: the tokenizer to use
        :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
            If set to true, this method returns a single int instead of a list and throws an error if the word
            corresponds to multiple tokens.
        :return: either the list of token ids or the single token id corresponding to this word
        """
        ids = self.tokenizer.encode(word, add_special_tokens=False, **self.kwargs)
        if not force_single_token:
            return ids
        assert len(ids) == 1, \
            f'Verbalization "{word}" does not correspond to a single token, got {self.tokenizer.convert_ids_to_tokens(ids)}'
        verbalization_id = ids[0]
        assert verbalization_id not in self.tokenizer.all_special_ids, \
            f'Verbalization {word} is mapped to a special token {self.tokenizer.convert_ids_to_tokens(verbalization_id)}'
        return verbalization_id


class DartDatasetSequenceClassification(torch.utils.data.Dataset):
    """Dataset for sequence classification tasks with DART"""

    def __init__(self, args, tokenizer, file_path, mode="train"):
        self.args = args
        self.task_name = args.downstream_task_name
        self.processor = processors_mapping[self.task_name.lower()]
        self.tokenizer = tokenizer
        self.kwargs = {'add_prefix_space': True} if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, RobertaTokenizer) else {}
        assert isinstance(self.tokenizer, RobertaTokenizer), "Only RobertaTokenizer is supported for now"
        self.mode = mode
        self.file_path = file_path
        self.max_length = args.max_seq_length

        # assert mode in ["train", "dev", "test", ""]

        # Get label list and its mapping to word
        self.label_to_word = map_of_mapping[self.task_name]
        self.word_to_label = {v: k for k, v in self.label_to_word.items()}
        self.label_map = {label: i for i, label in enumerate(self.label_to_word.keys())}

        self.pattern_index = []
        self.mlm_label_index = []

        # self.pattern = "In summary, it is"
        if self.task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', "aclImdb", "ag_news", "yelp_review", "yahoo_answers", "amazon_review", "reverse_SST-2"]:
            # For single sentence tasks
            self.pattern = "It was"
        elif self.task_name in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE', 'MNLI-mm']:
            # For pair sentence tasks
            self.pattern = ","

        # Resize model embedding to accommodate the new tokens
        # new_id = self.tokenizer.vocab_size
        if isinstance(self.tokenizer, RobertaTokenizer):
            new_id = 50265
        else:
            raise NotImplementedError

        # Set new id for pattern
        pattern_token_ids = self.enc(self.pattern)
        for origin_id in pattern_token_ids:
            self.pattern_index.append([origin_id, new_id])
            new_id += 1

        # Set new id for label
        self.word_to_new_label = {}
        for origin_word in self.word_to_label.keys():
            origin_label_id = self.enc(origin_word)
            assert len(origin_label_id) == 1
            origin_label_id = origin_label_id[0]
            self.mlm_label_index.append([origin_label_id, new_id])
            self.word_to_new_label[origin_word] = new_id
            new_id += 1

        # Add custom tokens, where tokenizer.vocab_size will not be affected
        if mode == "train":
            num_new_tokens = len(self.pattern_index) + len(self.mlm_label_index)
            custom_tokens = ['<my_token{}>'.format(str(i+1)) for i in range(num_new_tokens)]
            num_added_tokens = tokenizer.add_tokens(custom_tokens)
            logger.info('We have added {} tokens'.format(num_added_tokens))

        self.pattern_index_ids = [x[1] for x in self.pattern_index]
        self.mlm_logits_to_cls_logits_tensor = torch.tensor([x[1] for x in self.mlm_label_index], dtype=torch.long, requires_grad=False)

        # Load cache
        cache_path, file_name = os.path.split(self.file_path)
        file_name = file_name.split(".")[0]

        # cached_features_file = os.path.join(
        #     cache_path,
        #     "fine_tuning_cached_{}_{}_{}_{}".format(
        #         mode,
        #         tokenizer.__class__.__name__,
        #         str(args.max_seq_length),
        #         file_name,
        #     ),
        # )

        logger.info(f"Creating examples from dataset file at {self.file_path}")
        if self.mode == "train" and self.task_name in ["aclImdb", "ag_news", "yelp_review", "yahoo_answers", "amazon_review", "reverse_SST-2"]:
            self.examples = self.processor.get_train_examples(self.file_path, seed=self.args.seed, num_labelled_data=self.args.num_labelled_data)
        elif self.mode == "train":
            self.examples = self.processor.get_train_examples(self.file_path)
        elif self.mode == "dev":
            self.examples = self.processor.get_dev_examples(self.file_path)
        elif self.mode == "test":
            self.examples = self.processor.get_test_examples(self.file_path)
        elif "pseudo" in self.mode:
            if "train" in self.mode:
                self.examples = self.processor.get_psuedo_examples(os.path.join(self.file_path, "train.json"))
            elif "dev" in self.mode:
                self.examples = self.processor.get_psuedo_examples(os.path.join(self.file_path, "dev.json"))
        else:
            raise ValueError("Invalid mode: %s" % self.mode)

        logger.info(f"Creating {self.mode} features from dataset file at {self.file_path}")

        self.features = []

        for ex in tqdm.tqdm(self.examples, desc="Creating {} features".format(self.mode)):
                self.features.append(self.convert_fn(ex))

        logger.info("Getting {} {} samples in total.".format(len(self.features), self.mode))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return {"input_ids": self.features[i].input_ids,
                "attention_mask": self.features[i].attention_mask,
                "labels": self.features[i].label,
                "mlms": self.features[i].mlms}

    def enc(self, label):
        return self.tokenizer.encode(label, add_special_tokens=False, **self.kwargs)

    def convert_fn(self, example):
        """
        Returns a list of processed "InputFeatures".
        """
        
        input_ids, token_type_ids = self.encode(example)
        if self.task_name.lower() == "sts-b":
            # STS-B is a regression task
            label = float(example.label)
        else:
            label = self.label_map[example.label] if example.label is not None else None

        mlm_label = [-100] * len(input_ids)
        if self.task_name.lower() == "sts-b":
            # STS-B is a regression task
            hard_label = '0' if float(example.label) <= 2.5 else '1'
            new_label = self.word_to_new_label[self.label_to_word[hard_label]]
        else:        
            new_label = self.word_to_new_label[self.label_to_word[example.label]] if example.label is not None else 1
        mlm_index = input_ids.index(self.tokenizer.mask_token_id)
        mlm_label[mlm_index] = new_label
        attention_mask = [1] * len(input_ids)

        # Pad
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            mlm_label.append(-100)

        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        assert len(mlm_label) == self.max_length
        assert sum([1 if i > 0 else 0 for i in mlm_label])
        
        return OurInputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label,
            mlms=mlm_label,
        )
 
    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        processed_parts_a = []
        for x, s in parts_a:
            if not x:
                continue
            if x == 'PLACEHOLDER':
                processed_parts_a.append((self.pattern_index_ids, s))
            else:
                processed_parts_a.append((self.enc(x), s))
        
        processed_parts_b = []
        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            for x, s in parts_b:
                if not x:
                    continue
                if x == 'PLACEHOLDER':
                    processed_parts_b.append((self.pattern_index_ids, s))
                else:
                    processed_parts_b.append((self.enc(x), s))

        num_special = self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(processed_parts_a, processed_parts_b, max_length=self.max_length - num_special)

        tokens_a = [token_id for part, _ in processed_parts_a for token_id in part]
        tokens_b = [token_id for part, _ in processed_parts_b for token_id in part] if processed_parts_b else []

        if tokens_b:
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        else:
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a)

        return input_ids, token_type_ids

    def get_parts(self, example: InputExample):
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        if self.task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', "aclImdb", "ag_news", "yelp_review", "yahoo_answers", "amazon_review", "reverse_SST-2"]:
            # Single Sentence Tasks
            return [text_a, text_b, 'PLACEHOLDER', self.tokenizer.mask_token], []
        elif self.task_name in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE', 'MNLI-mm']:
            # Sentence Pair Tasks
            return [text_a, self.tokenizer.mask_token, 'PLACEHOLDER', text_b], []

    def shortenable(self, s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True
    
    def _seq_length(self, parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    def _remove_last(self, parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)
                
                
class CLSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.label = data_dict["label"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.label[index]
        }