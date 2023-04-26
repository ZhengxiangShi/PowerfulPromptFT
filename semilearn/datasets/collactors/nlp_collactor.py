# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import GPT2Tokenizer, RobertaTokenizer
from transformers import BertTokenizerFast, RobertaTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data import default_data_collator


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        self.kwargs = {'add_prefix_space': True} if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, RobertaTokenizer) else {}
        w_features = []
        s_features_ = []
        s_features = []
        for f in features:
            f_ = {k:v for k,v in f.items() if 'text' not in k}
            # input_ids = self.tokenizer(f['text'], max_length=self.max_length, truncation=True, padding=False)['input_ids']
            input_ids, token_type_ids = self.encode(f['text'])
            f_['input_ids'] = input_ids 
            mlm_label = [-100] * len(input_ids)
            mlm_index = input_ids.index(self.tokenizer.mask_token_id)
            # mlm_label[mlm_index] = self.tokenizer.encode(label_to_word[label], add_special_tokens=False, **self.kwargs)[0]
            mlm_label[mlm_index] = 1
            attention_mask = [1] * len(input_ids)       
            while len(input_ids) < self.max_length:
                input_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)
                mlm_label.append(-100)            
            f_["attention_mask"] = attention_mask
            f_["mlms"] = mlm_label
            w_features.append(f_)

            if 'text_s' in f:
                # input_ids_s = self.tokenizer(f['text_s'], max_length=self.max_length, truncation=True, padding=False)['input_ids']
                input_ids_s, _ = self.encode(f['text_s'])
                mlm_label = [-100] * len(input_ids_s)
                mlm_index = input_ids_s.index(self.tokenizer.mask_token_id)
                # mlm_label[mlm_index] = self.tokenizer.encode(label_to_word[label], add_special_tokens=False, **self.kwargs)[0]
                mlm_label[mlm_index] = 1
                attention_mask = [1] * len(input_ids_s)       
                while len(input_ids_s) < self.max_length:
                    input_ids_s.append(self.tokenizer.pad_token_id)
                    attention_mask.append(0)
                    mlm_label.append(-100)            
                s_features.append({'input_ids':input_ids_s, 'attention_mask':attention_mask, 'mlms':mlm_label})

            if 'text_s_' in f:
                # input_ids_s_ = self.tokenizer(f['text_s_'], max_length=self.max_length, truncation=True, padding=False)['input_ids']
                input_ids_s_, _ = self.encode(f['text_s_'])
                mlm_label = [-100] * len(input_ids_s_)
                mlm_index = input_ids_s_.index(self.tokenizer.mask_token_id)
                # mlm_label[mlm_index] = self.tokenizer.encode(label_to_word[label], add_special_tokens=False, **self.kwargs)[0]
                mlm_label[mlm_index] = 1
                attention_mask = [1] * len(input_ids_s_)       
                while len(input_ids_s_) < self.max_length:
                    input_ids_s_.append(self.tokenizer.pad_token_id)
                    attention_mask.append(0)
                    mlm_label.append(-100) 
                s_features_.append({'input_ids':input_ids_s_, 'attention_mask':attention_mask, 'mlms': mlm_label})
        
        batch = self.tokenizer.pad(
            w_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        if 'label' in batch:
            # return {'idx_lb': batch['idx'], 'x_lb': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, 'y_lb': batch['label']}
            return {'idx_lb': batch['idx'], 'x_lb': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'mlms': batch['mlms']}, 'y_lb': batch['label']}
        else:
            if len(s_features) > 0:
                s_batch = self.tokenizer.pad(
                    s_features,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                if len(s_features_) > 0:
                    s_batch_ = self.tokenizer.pad(
                        s_features_,
                        padding=True,
                        max_length=None,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    )
                    return {'idx_ulb': batch['idx'], 
                            'x_ulb_w': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'mlms': batch['mlms']}, \
                            'x_ulb_s_0': {'input_ids': s_batch['input_ids'], 'attention_mask': s_batch['attention_mask'], 'mlms': batch['mlms']}, \
                            'x_ulb_s_1': {'input_ids': s_batch_['input_ids'], 'attention_mask': s_batch_['attention_mask'], 'mlms': batch['mlms']}
                        }
                else:
                    return {'idx_ulb': batch['idx'], 'x_ulb_w': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'mlms': batch['mlms']},
                                                     'x_ulb_s': {'input_ids': s_batch['input_ids'], 'attention_mask': s_batch['attention_mask'], 'mlms': batch['mlms']}}
            else:
                return {'idx_ulb': batch['idx'], 'x_ulb_w': {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'mlms': batch['mlms']}}


    def encode(self, example) -> Tuple[List[int], List[int]]:
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
    
    def get_parts(self, text_a):
        text_b = None # Only support single sentence tasks
        text_a = self.shortenable(text_a)
        text_b = self.shortenable(text_b)
        return aclimdb_template(text_a, text_b, self.tokenizer)

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

def aclimdb_template(text_a, text_b, tokenizer):
    return [text_a, text_b, "It was", tokenizer.mask_token, "."], []

template_mapping = {
    "aclImdb": aclimdb_template,
    "ag_news": aclimdb_template,
    "amazon_review": aclimdb_template,
    "yahoo_answers": aclimdb_template,
    "yelp_review": aclimdb_template,
}

def get_bert_base_uncased_collactor(max_length=512):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn


def get_bert_base_cased_collactor(max_length=512):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn


def get_roberta_base_collactor(max_length=512):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn

def get_roberta_large_collactor(max_length=512):
    print("Loading roberta-large tokenizer, with max_length={}".format(max_length))
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn