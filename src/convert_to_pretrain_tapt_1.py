"""
This script converts the original data to the format used in the pretraining script.
This is for task adaptive pre-training (conventional continued pre-training) without using the pseudo labels.
The generated json files are used for pseudo label generation in the next step.
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from processors import processors_mapping
from transformers import InputExample


def get_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'SST-2':
            return line[-1]
        elif task == 'STS-B':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style (tsv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "MNLI":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            # Other datasets (csv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets

def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    parser.add_argument("--task", type=str,
        # default=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'],
        default="double",
        help="Task names")
    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data/glue_pretrain_full", help="Output path")
    args = parser.parse_args()
    if args.task == "single":
        args.task = ['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA']
    elif args.task == "double":
        args.task = ['MNLI', 'SNLI', 'QNLI', 'QQP']
        # args.task = ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE']
    else:
        raise ValueError("Unknown task type.")
    datasets = load_datasets(args.data_dir, args.task)

    os.makedirs(args.output_dir, exist_ok=True)
    for task, dataset in datasets.items():
        # Shuffle the training set
        print("=========================================")
        print("| Task = %s" % (task))
        if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style 
            train_header, train_lines = split_header(task, dataset["train"])
            np.random.shuffle(train_lines)
        else:
            # Other datasets 
            train_lines = dataset['train'].values.tolist()

        # if task in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'] and len(train_lines) > 10000:
        #     # We only use 10k examples for MRPC, QQP, STS-B, MNLI, SNLI, QNLI, RTE
        #     np.random.shuffle(train_lines)
        #     train_lines = train_lines[:10000]

        os.makedirs(os.path.join(args.output_dir, task), exist_ok=True)
        if task in ["SST-2", "CoLA", 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE']:
            train_lines = [line.strip('\n').split('\t') for line in train_lines]
        train_examples = processors_mapping[task.lower()]._create_examples(train_lines, "train")

        print("| Train size = %d" % (len(train_examples)))
        train_data, dev_data = train_test_split(train_examples, train_size=0.9, random_state=42)
        print("| Train size = %d | Dev size = %d" % (len(train_data), len(dev_data)))

        lengths = []
        if task in ['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA']:
            for split, data in zip(['train', 'dev'], [train_data, dev_data]):
                with open(os.path.join(args.output_dir, task, f'{split}.txt'), 'w') as f, open(os.path.join(args.output_dir, task, f'{split}.json'), 'w') as f_json:
                    output_examples = {}
                    count = 0
                    for input_example in data:
                        try:
                            f.write(input_example.text_a + '\n')
                            lengths.append(len(input_example.text_a.split()))
                            output_examples[str(count)] = {"text_a": input_example.text_a, 
                                                            "text_b": input_example.text_b,
                                                            "label": input_example.label}
                            count += 1
                        except TypeError:
                            print("| Skipping one example for task %s" % task)
                    json.dump(output_examples, f_json, indent=4)
        elif task in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE']:
            for split, data in zip(['train', 'dev'], [train_data, dev_data]):
                with open(os.path.join(args.output_dir, task, f'{split}.txt'), 'w') as f, open(os.path.join(args.output_dir, task, f'{split}.json'), 'w') as f_json:
                    output_examples = {}
                    count = 0
                    for input_example in data:
                        try:
                            f.write(input_example.text_a + " " + input_example.text_b + '\n')
                            lengths.append(len(input_example.text_a.split()) + len(input_example.text_b.split()) )
                            output_examples[str(count)] = {"text_a": input_example.text_a, 
                                                            "text_b": input_example.text_b,
                                                            "label": input_example.label}
                            count += 1
                        except TypeError:
                            print("| Skipping one example for task %s" % task)
                    json.dump(output_examples, f_json, indent=4)
        else:
            raise NotImplementedError
        print("| Avg length = %f" % (sum(lengths) / len(lengths)))
    print("=========================================")


if __name__ == "__main__":
    main()
