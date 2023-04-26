"""This script converts the full train and dev dataset with generated pseudo labels to the pretrain semi-prompt format."""
import os
import json
import tqdm
import argparse
# from src.ssl_processors import processors_mapping


def ssl_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "It was", label, "."]

def dart_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "It was", label]

def dart_ts_template(text_a: str, text_b: str, label: str):
    return [text_a, label, ",", text_b]

# One sentence tasks
def sst2_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "It was", label, "."] 

def sst5_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "It was", label, "."] 

def mr_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "It was", label, "."] 

def cr_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "It was", label, "."] 

def mpqa_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "is", label, "."] 

def subj_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "This is", label, "."] 

def trec_template(text_a: str, text_b: str, label: str):
    return [label, ":", text_a, text_b] 

def cola_template(text_a: str, text_b: str, label: str):
    return [text_a, text_b, "This is", label] 

# Two sentence tasks
def mrpc_template(text_a: str, text_b: str, label: str):
    return [text_a, label, ",", text_b]

def qqp_template(text_a: str, text_b: str, label: str):
    return [text_a, label, ",", text_b]

def sstb_template(text_a: str, text_b: str, label: str):
    return [text_a, label, ",", text_b]

def mnli_template(text_a: str, text_b: str, label: str):
    return [text_a, "?", label, ",", text_b]

def snli_template(text_a: str, text_b: str, label: str):
    return [text_a, ".", label, ", in this case", text_b]

def qnli_template(text_a: str, text_b: str, label: str):
    return [text_a, "?", label, ",", text_b]

def rte_template(text_a: str, text_b: str, label: str):
    return [text_a, ".", label, ", I think that", text_b]


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
    'RTE': {'not_entailment':'Yet', 'entailment':'Clearly'},

    "aclImdb": {"0": "great", "1": "terrible"},
    "reverse_SST-2": {"0": "great", "1": "terrible"},
    "ag_news": {"0": "world", "1": "sports", "2": "business", "3": "tech"},
    "yelp_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    "amazon_review": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    "yahoo_answers": {"0": "culture", "1": "science", "2": "health", "3": "education", "4": "computer", "5": "sports", "6": "business", "7": "music", "8": "family", "9": "politics"},
}

# for k, v in map_of_mapping.items():
#     map_of_mapping[k] = {str(k): v for k, v in v.items()}
    
for k, v in map_of_mapping.items():
    map_of_mapping[k] = {str(i): v for i, (k, v) in enumerate(v.items())}
    

def format_dataset(task_name=None, train_file=None, dev_file=None, output_path=None, confidence=0.0, num_sample=1, use_fixed_dart=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if num_sample != 1:
        raise NotImplementedError("Only support num_sample=1.")

    output_train_file = os.path.join(output_path, 'train.txt')
    output_dev_file = os.path.join(output_path, 'dev.txt')

    for input_file, output_file in zip([train_file, dev_file], [output_train_file, output_dev_file]):
        print("Processing {} to {}".format(input_file, output_file))
        with open(input_file, 'r') as f, open(output_file, 'w') as f_output:
            data = json.load(f)

            for _, doc_item in tqdm.tqdm(data.items()):
                text_a = doc_item["text_a"]
                text_b = doc_item["text_b"]
                pred = doc_item["pred"]
                label = map_of_mapping[task_name][pred]

                if use_fixed_dart:
                    if task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', "aclImdb", "ag_news", "yelp_review", "yahoo_answers", "amazon_review", "reverse_SST-2"]:
                        new_text = dart_template(text_a, text_b, label)
                    elif task_name in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'MNLI-mm', 'SNLI', 'QNLI', 'RTE']:
                        new_text = dart_ts_template(text_a, text_b, label)
                    else:
                        raise NotImplementedError("Not implemented for task {}".format(task_name))
                else:
                    new_text = template_mapping[task_name](text_a, text_b, label)
                new_text = [x for x in new_text if x]
                f_output.write(' '.join(new_text) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="The name of the task."
    )
    parser.add_argument(
        "--train_file", default=None, type=str, required=True, help="The file path of the train data."
    )
    parser.add_argument(
        "--dev_file", default=None, type=str, required=True, help="The file path of the dev data."
    )
    parser.add_argument(
        "--output_path", default="data", type=str, help="The output path of train and dev files."
    )
    parser.add_argument(
        "--confidence", default=0.0, type=float, help="The confidence threshold for selecting samples."
    )
    parser.add_argument(
        "--num_sample", default=1, type=int, help="The number of samples in each train sample."
    )
    parser.add_argument(
        "--use_fixed_dart", action='store_true', default=False, help="Use fixed dart template."
    )
    args = parser.parse_args()
    format_dataset(args.task_name, args.train_file, args.dev_file, args.output_path, args.confidence, args.num_sample, args.use_fixed_dart)
