# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.activations import gelu


class ClassificationRoberta(nn.Module):
    def __init__(self, name, num_classes=2):
        super(ClassificationRoberta, self).__init__()
        # Load pre-trained bert model
        self.roberta = RobertaModel.from_pretrained(name)  # "roberta-base"
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        print(name)
        self.num_features = 1024 if "large" in name else 768
        self.classifier = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.GELU(),
            nn.Linear(self.num_features, num_classes)
        ])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            return_embed: return word embedding, used for vat
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        
        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)

        # Method 1
        pooled_output = self.dropout(out_dict['pooler_output'])

        # Method 2
        # last_hidden = out_dict['last_hidden_state']
        # drop_hidden = self.dropout(last_hidden)
        # pooled_output = torch.mean(drop_hidden, 1)

        if only_feat:
            return pooled_output
        
        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
            
        return result_dict
        
    def extract(self, x):
        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}roberta.embeddings'.format(prefix), blocks=r'^{}roberta.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []


class RobertaForPromptingClassification(nn.Module):
    """Roberta model for classification with prompting."""

    def __init__(self, name, num_classes=2):
        super(RobertaForPromptingClassification, self).__init__()

        self.roberta = RobertaModel.from_pretrained(name)
        self.lm_head = RobertaLMHead()
        
        self.mlm_logits_to_cls_logits_tensor = None
        self.num_labels = None

        # For regression
        self.lb = None
        self.ub = None

    def extract(self, x):
        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}roberta.embeddings'.format(prefix), blocks=r'^{}roberta.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []

    def forward(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        mlms = x['mlms']

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_logits = prediction_scores[mlms >= 0]  # [batch_size, vocab_size]
        logits = torch.index_select(masked_logits, 1, self.mlm_logits_to_cls_logits_tensor.to(masked_logits.device))  # [batch_size, num_labels]
        result_dict = {'logits':logits, 'feat': masked_logits}

        return result_dict


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, hidden_size=1024, layer_norm_eps=1e-05, vocab_size=50265):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


# def roberta_large(pretrained=True, pretrained_path=None, **kwargs):
#     if not pretrained_path:
#         # pretrained_path = 'roberta-base'
#         pretrained_path = 'roberta-large'
#     print('Loading pretrained model: {}'.format(pretrained_path))
#     model = ClassificationRoberta(name=pretrained_path, **kwargs)
#     return model

def roberta_base(pretrained=True, pretrained_path=None, **kwargs):
    if not pretrained_path:
        pretrained_path = 'roberta-base'
        # pretrained_path = 'roberta-large'
    print('Loading pretrained model: {}'.format(pretrained_path))
    model = ClassificationRoberta(name=pretrained_path, **kwargs)
    return model

def roberta_for_prompting_classification(pretrained=True, pretrained_path=None, **kwargs):
    if not pretrained_path:
        pretrained_path = 'roberta-large'
    print('Loading pretrained model: {}'.format(pretrained_path))
    model = RobertaForPromptingClassification(name=pretrained_path, **kwargs)
    return model



label_to_word = {
    "yahoo_answers": {"0": "culture", "1": "science", "2": "health", "3": "education", "4": "computer", "5": "sports", "6": "business", "7": "music", "8": "family", "9": "politics"},
    "amazon_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    "yelp_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    "ag_news": {"0": "world", "1": "sports", "2": "business", "3": "tech"},
    "aclImdb": {"0": "great", "1": "terrible"},
}