import argparse
import re
import tqdm
import os
import math
import nltk
import numpy as np
import string
import torch

#from nltk.corpus import wordnet
#from datasets import load_dataset
#from nltk.tokenize.treebank import TreebankWordDetokenizer
#from tenacity import (
#    retry,
#    stop_after_attempt,
#    wait_random_exponential,
#)
#from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from common.ghostbusters_load import Dataset, get_generate_dataset
from common.ghostbusters_write_logprobs import write_logprobs_gpt2
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

if __name__ == "__main__":
    if args.logprobs:
        datasets = [
            Dataset("normal", "data/wp/human"),
            Dataset("normal", "data/wp/gpt"),
            Dataset("author", "data/reuter/human"),
            Dataset("author", "data/reuter/gpt"),
            Dataset("normal", "data/essay/human"),
            Dataset("normal", "data/essay/gpt"),
        ]
        generate_logprobs(get_generate_dataset(*datasets))

    if args.logprob_other:
        other_datasets = [
            Dataset("normal", "data/other/ets"),
            Dataset("normal", "data/other/lang8"),
            Dataset("normal", "data/other/pelic"),
            Dataset("normal", "data/other/gptzero/gpt"),
            Dataset("normal", "data/other/gptzero/human"),
            Dataset("normal", "data/other/toefl91"),
            Dataset("normal", "data/other/undetectable"),
        ]

        generate_logprobs(get_generate_dataset(*other_datasets))