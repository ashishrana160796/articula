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

trigram_model_path = "models/trigram_model.pkl"
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

eval_dataset = [
    Dataset("normal", "data/wp/claude"),
    Dataset("author", "data/reuter/claude"),
    Dataset("normal", "data/essay/claude"),
    Dataset("normal", "data/wp/gpt_prompt1"),
    Dataset("author", "data/reuter/gpt_prompt1"),
    Dataset("normal", "data/essay/gpt_prompt1"),
    Dataset("normal", "data/wp/gpt_semantic"),
    Dataset("author", "data/reuter/gpt_semantic"),
    Dataset("normal", "data/essay/gpt_semantic"),
]


def get_featurized_data(generate_dataset_fn, best_features):
    t_data = generate_dataset_fn(t_featurize)

    gpt2, trigram, unigram = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer
    )

    vector_map = {
        "gpt2-logprobs": lambda file: gpt2[file],
        "trigram-logprobs": lambda file: trigram[file],
        "unigram-logprobs": lambda file: unigram[file],
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data = generate_dataset_fn(exp_featurize)

    return np.concatenate([t_data, exp_data], axis=1)

parser = argparse.ArgumentParser()
parser.add_argument("--generate_symbolic_data", action="store_true")
parser.add_argument("--generate_symbolic_data_eval", action="store_true")

parser.add_argument("--perform_feature_selection", action="store_true")
#parser.add_argument("--perform_feature_selection_one", action="store_true")
#parser.add_argument("--perform_feature_selection_two", action="store_true")
#parser.add_argument("--perform_feature_selection_four", action="store_true")

parser.add_argument("--perform_feature_selection_domain", action="store_true")

parser.add_argument("--train_on_all_data", action="store_true")
parser.add_argument("--seed", type=int, default=2024)
args = parser.parse_args()

np.random.seed(args.seed)

if __name__ == "__main__":
    result_table = [["F1", "Accuracy", "AUC"]]

    datasets = [
        *reddit_dataset,
        *wikip_dataset]
    
    generate_dataset_fn = get_generate_dataset(*datasets)

    if args.generate_symbolic_data:
        generate_symbolic_data(generate_dataset_fn,
                               max_depth=3,
                               trigram = trigram_model,
                               tokenizer = tokenizer,
                               output_file="symbolic_data_gpt",
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
        exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train
        )

        with open("results/gb_best_features_three.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_domain:
        exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))

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
    
    data, mu, sigma = normalize(
        get_featurized_data(generate_dataset_fn, best_features), ret_mu_sigma=True
    )

    print(f"Best Features: {best_features}")
    print(f"Data Shape: {data.shape}")

    base = LogisticRegression()
    model = CalibratedClassifierCV(base, cv=5)

    if args.train_on_all_data:
        model.fit(data, labels)

        with open("models/gb_features.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")
        pickle.dump(model, open("models/gb_model", "wb"))
        pickle.dump(mu, open("models/gb_mu", "wb"))
        pickle.dump(sigma, open("models/gb_sigma", "wb"))
        pickle.dump(trigram_model, open(trigram_model_path, "wb"))

        print("Saved model to model/")
    else:
        model.fit(data[train], labels[train])

    predictions = model.predict(data[test])
    probs = model.predict_proba(data[test])[:, 1]

    result_table.append(
        [
            round(f1_score(labels[test], predictions), 3),
            round(accuracy_score(labels[test], predictions), 3),
            round(roc_auc_score(labels[test], probs), 3),
        ]
    )

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))