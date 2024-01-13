import os
import argparse
import math
import numpy as np
import tiktoken
import dill as pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from tabulate import tabulate

from common.ghostbusters_featurize import normalize, t_featurize, select_features
from common.ghostbusters_symbolic import get_all_logprobs, train_trigram, get_exp_featurize
from common.ghostbusters_symbolic import generate_symbolic_data
from common.ghostbusters_load import get_generate_dataset, Dataset

best_features_path = "results/gb_best_features_three.txt" 
if os.path.exists(best_features_path):
    with open(best_features_path) as f:
        best_features = f.read().strip().split("\n")
else:
    print("Need to peform feature selection")

trigram_model_path = "models/trigram_model_mling.pkl"
if not os.path.exists(trigram_model_path):
    print("Training trigram model...")
    trigram_model, tokenizer = train_trigram(verbose=True, return_tokenizer=True)
else:
    print("Loading trigram model...")
    trigram_model = pickle.load(
        open(trigram_model_path, "rb"), pickle.HIGHEST_PROTOCOL)
    tokenizer = tiktoken.encoding_for_model("davinci").encode

reddit_dataset = [
    Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/reddit/human"),
    Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/reddit/ai"),
]

wikip_dataset = [
    Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/wikip/human"),
    Dataset("normal", "data/transformed-model-input-datasets/intrinsic_dim_data/wikip/ai"),
]

generated_dataset_small = [
    Dataset("normal", "data/transformed-model-input-datasets/generated_gpt2_en_data/ai"),
    Dataset("normal", "data/transformed-model-input-datasets/generated_gpt2_de_data_small/ai")
]

generated_dataset_large = [
    Dataset("normal", "data/transformed-model-input-datasets/generated_gpt2_de_data_large/ai")
]

eval_dataset = [
    Dataset("normal", "data/toefl91"),
    Dataset("normal", "data/informaticup-test-dataset/informaticup-dataset/ai_gen_text")
]

def get_featurized_data(generate_dataset_fn, best_features):
    t_data = generate_dataset_fn(t_featurize)

    gpt2, db, trigram, unigram = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer
    )

    vector_map = {
        "gpt2-logprobs": lambda file: gpt2[file],
        "db-logprobs": lambda file: db[file],
        "trigram-logprobs": lambda file: trigram[file],
        "unigram-logprobs": lambda file: unigram[file],
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data = generate_dataset_fn(exp_featurize)

    return np.concatenate([t_data, exp_data], axis=1)

parser = argparse.ArgumentParser()
parser.add_argument("--generate_symbolic_data", action="store_true")
parser.add_argument("--generate_symbolic_data_four", action="store_true")
parser.add_argument("--generate_symbolic_data_eval", action="store_true")

parser.add_argument("--perform_feature_selection", action="store_true")
parser.add_argument("--perform_feature_selection_four", action="store_true")
parser.add_argument("--perform_feature_selection_large", action="store_true")
parser.add_argument("--perform_feature_selection_lang", action="store_true")
parser.add_argument("--perform_feature_selection_domain", action="store_true")

parser.add_argument("--train_on_all_data", action="store_true")
parser.add_argument("--seed", type=int, default=2024)
args = parser.parse_args()

np.random.seed(args.seed)

if __name__ == "__main__":
    result_table = [["F1", "Accuracy", "AUC"]]

    datasets = [*reddit_dataset,
                *wikip_dataset,
                *generated_dataset_small,
                *generated_dataset_large]
    
    generate_dataset_fn = get_generate_dataset(*datasets)

    if args.generate_symbolic_data:
        generate_symbolic_data(generate_dataset_fn,
                               max_depth = 3,
                               trigram = trigram_model,
                               tokenizer = tokenizer,
                               output_file = "symbolic_data_gpt_db",
                               verbose = True)
        
        t_data = generate_dataset_fn(t_featurize)
        pickle.dump(t_data, open("t_data", "wb"))
    
    if args.generate_symbolic_data_four:
        generate_symbolic_data(generate_dataset_fn,
                               max_depth=4,
                               output_file="symbolic_data_gpt_db_four",
                               verbose=True)

        t_data = generate_dataset_fn(t_featurize)
        pickle.dump(t_data, open("t_data", "wb"))
    
    if args.generate_symbolic_data_eval:
        generate_dataset_fn_eval = get_generate_dataset(*eval_dataset)
        
        generate_symbolic_data(generate_dataset_fn_eval,
                               max_depth=3,
                               trigram = trigram_model,
                               tokenizer=tokenizer,
                               output_file="symbolic_data_eval",
                               verbose=True)
        
        t_data_eval = generate_dataset_fn_eval(t_featurize)
        pickle.dump(t_data_eval, open("t_data_eval", "wb"))
    
    labels = generate_dataset_fn(lambda file: 1 if any([m in file for m in ["ai"]]) else 0)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :])
    
    print("Train/Test Split", train, test)
    print("Train Size:", len(train), "Valid Size:", len(test))
    print(f"Positive Labels: {sum(labels[indices])}, Total Labels: {len(indices)}")

    if args.perform_feature_selection:
        print("Peforming Feature Selection...")
        exp_to_data = pickle.load(open("symbolic_data_gpt_db", "rb"))
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train
        )

        with open("results/gb_best_features_three.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_four:
        print("Peforming Feature Selection...")
        exp_to_data = pickle.load(open("symbolic_data_gpt_db_four", "rb"))
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train
        )

        with open("results/gb_best_features_four.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")
    
    if args.perform_feature_selection_large:
        exp_to_data = pickle.load(open("symbolic_data_gpt_db", "rb"))

        large_indices = np.where(generate_dataset_fn(lambda file: "generated_gpt2_en_data" or "generated_gpt2_de_data_small" not in file))[0]
        
        large_features = select_features(
            exp_to_data, labels, verbose = True, to_normalize = True, indices=large_indices
        )
        with open("results/gb_best_features_large.txt", "w") as f:
            for feat in large_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_lang:
        exp_to_data = pickle.load(open("symbolic_data_gpt_db", "rb"))

        en_indices = np.where(generate_dataset_fn(lambda file: "_en_" in file))[0]
        de_indices = np.where(generate_dataset_fn(lambda file: "_de_" in file))[0]

        en_features = select_features(
            exp_to_data, labels, verbose = True, to_normalize = True, indices=en_indices
        )
        with open("results/gb_best_features_en.txt", "w") as f:
            for feat in en_features:
                f.write(feat + "\n")

        de_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=de_indices
        )
        with open("results/gb_best_features_de.txt", "w") as f:
            for feat in de_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_domain:
        exp_to_data = pickle.load(open("symbolic_data_gpt_db", "rb"))

        reddit_indices = np.where(generate_dataset_fn(lambda file: "reddit" in file))[0]
        wikip_indices = np.where(generate_dataset_fn(lambda file: "wikip" in file))[0]

        reddit_features = select_features(
            exp_to_data, labels, verbose = True, to_normalize = True, indices=reddit_indices
        )
        with open("results/gb_best_features_reddit.txt", "w") as f:
            for feat in reddit_features:
                f.write(feat + "\n")

        wikip_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=wikip_indices
        )
        with open("results/gb_best_features_wikip.txt", "w") as f:
            for feat in wikip_features:
                f.write(feat + "\n")