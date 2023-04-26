
""" 
This script converts the datasets (aclImdb ag_news amazon_review yahoo_answers yelp_review)
into the format that can be used for pretraining (masked language modelling).
"""

import os
import json
import tqdm
import argparse

def format_dataset(input_path):
    train_file = os.path.join(input_path, 'train.json')
    dev_file = os.path.join(input_path, 'dev.json')
    test_file = os.path.join(input_path, 'test.json')

    output_train_file = os.path.join(input_path, 'train.txt')
    output_dev_file = os.path.join(input_path, 'dev.txt')
    output_test_file = os.path.join(input_path, 'test.txt')
    
    for input_file, output_file in zip([train_file, dev_file, test_file], [output_train_file, output_dev_file, output_test_file]):
        with open(input_file, 'r') as f:
            data = json.load(f)
        print('Processing {} into {}.'.format(input_file, output_file))
        with open(output_file, 'w') as f:
            for doc_id, doc_item in tqdm.tqdm(data.items()):
                f.write(doc_item['ori'] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", default="data", type=str, help="The path of train, dev, and test files."
    )
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="The name of the task."
    )
    args = parser.parse_args()
    format_dataset(os.path.join(args.file_path, args.task_name))
