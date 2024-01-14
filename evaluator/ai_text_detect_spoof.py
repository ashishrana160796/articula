import os
import argparse
import numpy as np
import dill as pickle

from evaluator.feature_generator import gb_mle_combined
from common.ghostbusters_featurize import normalize
from spoofer.baseline_text_spoofer import BaselineTextSpoofer

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='logistic')
args = parser.parse_args()

# Load model
model = pickle.load(open("models/comb_model_rf", "rb"))
mu = pickle.load(open("models/comb_mu", "rb"))
sigma = pickle.load(open("models/comb_sigma", "rb"))

def text_detector(text):

    features = gb_mle_combined(text)
    normalized_features = (features - mu) / sigma

    preds = model.predict_proba(normalized_features.reshape(-1, 1).T)[:, 1]

    if preds >= 0.5:
        print("The text is AI Generated!!! Spoofing...")
        text_spoofer = BaselineTextSpoofer(text, add_info_mutation=True)
        spoofed_text = text_spoofer.spoof_text()
        print(spoofed_text)
    else:
        print("Text is Human Written!")


if __name__ == '__main__':
    text_detector(args.input)