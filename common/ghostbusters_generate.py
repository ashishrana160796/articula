import argparse
import re
import tqdm
import os
import math
import nltk
import numpy as np
import string
import torch

from common.ghostbusters_load import Dataset, get_generate_dataset
from common.ghostbusters_write_logprobs import write_logprobs_gpt2, write_logprobs_db, write_logprobs_xlmr
from common.ghostbusters_symbolic import convert_file_to_logprob_file

parser = argparse.ArgumentParser()

parser.add_argument("--logprobs", action="store_true")
parser.add_argument("--logprob_other", action="store_true")
args = parser.parse_args()

def generate_logprobs(generate_dataset_fn):
    files = generate_dataset_fn(lambda f: f)

    for file in tqdm.tqdm(files):
        base_path = os.path.dirname(file) + "/logprobs"
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        with open(file, "r") as f:
            doc = f.read().strip()

        gpt2_file = convert_file_to_logprob_file(file, "gpt2")
        if not os.path.exists(gpt2_file):
            write_logprobs_gpt2(doc, gpt2_file)

        db_file = convert_file_to_logprob_file(file, "db")
        if not os.path.exists(db_file):
            write_logprobs_db(doc, db_file)
        
        

if __name__ == "__main__":
    if args.logprobs:
        datasets = [
            Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/reddit/human"),
            Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/reddit/ai"),
            Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/wikip/human"),
            Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/wikip/ai"),
            Dataset("normal", "data/transformed-model-input-datasets/generated_gpt2_en_data/ai"),
            Dataset("normal", "data/transformed-model-input-datasets/generated_gpt2_de_data_small/ai"),
            Dataset("normal", "data/transformed-model-input-datasets/generated_gpt2_de_data_large/ai")
        ]
        generate_logprobs(get_generate_dataset(*datasets))

    if args.logprob_other:
        other_datasets = [
            Dataset("normal", "data/toefl91"),
            Dataset("normal", "data/informaticup-test-dataset/informaticup-dataset/ai_gen_text")
        ]

        generate_logprobs(get_generate_dataset(*other_datasets))