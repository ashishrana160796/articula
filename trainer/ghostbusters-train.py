import os
import argparse
import math
import numpy as np
import tiktoken
import dill as pickle

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from tabulate import tabulate

from evaluator.feature_generator import gb_mle_combined_file, get_intrinsic_mle_file
from common.ghostbusters_featurize import normalize, t_featurize
from common.ghostbusters_symbolic import get_all_logprobs, train_trigram, get_exp_featurize
from common.ghostbusters_load import get_generate_dataset, Dataset


symbolic_data_path = "symbolic_data_gpt_db"
if os.path.exists(symbolic_data_path):
    print("Loading features...")
    exp_to_data = pickle.load(open(symbolic_data_path, "rb"))
    t_data = pickle.load(open("t_data", "rb"))
else:
    print("Need generate Symbolic Data to get features")

best_features_path = "models/gb_features_all.txt" 
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

def get_featurized_data(best_features):
    gpt_data = np.concatenate([t_data] + [exp_to_data[i] for i in best_features], axis=1)
    return gpt_data

def get_data(generate_dataset_fn, best_features):
    gpt2, db, trigram, unigram = get_all_logprobs(
          generate_dataset_fn,
          trigram=trigram_model,
          tokenizer=tokenizer,
          )
    vector_map = {
          "gpt2-logprobs": lambda file: gpt2[file],
          "db-logprobs": lambda file: db[file],
          "trigram-logprobs": lambda file: trigram[file],
          "unigram-logprobs": lambda file: unigram[file],
          }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data = generate_dataset_fn(exp_featurize)

    t_data = generate_dataset_fn(t_featurize, verbose=True)

    return np.concatenate([t_data, exp_data], axis=1)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, choices=['logistic', 'xgboost', 'random_forest'], default='logistic')
parser.add_argument("--ghostbuster", action="store_true")
parser.add_argument("--mle", action="store_true")
parser.add_argument("--combined", action="store_true")
parser.add_argument("--seed", type=int, default=2024)
args = parser.parse_args()

np.random.seed(args.seed)

if __name__ == "__main__":
    result_table = [["Dataset","F1", "Accuracy", "AUC"]]

    datasets = [*reddit_dataset,
                *wikip_dataset,
                *generated_dataset_small,
                *generated_dataset_large]
    
    generate_dataset_fn = get_generate_dataset(*datasets)
    
    if args.ghostbuster:
        #Training data
        data = get_featurized_data(best_features)
        data, mu, sigma = normalize(data, ret_mu_sigma=True)
        pickle.dump(mu, open("models/gb_mu", "wb"))
        pickle.dump(sigma, open("models/gb_sigma", "wb"))
        labels = generate_dataset_fn(lambda file: 1 if any([m in file for m in ["ai"]]) else 0)

        #Testing data
        toefl = get_generate_dataset(Dataset("normal", "data/toefl91"))
        toefl_labels = toefl(lambda _: 0)
        toefl_data = get_data(toefl, best_features)
        toefl_data = normalize(toefl_data, mu=mu, sigma=sigma)

        informaticup = get_generate_dataset(Dataset("normal", "data/informaticup-test-dataset/informaticup-dataset/ai_gen_text"))
        informaticup_labels = informaticup(lambda _: 1)
        informaticup_data = get_data(informaticup, best_features)
        informaticup_data = normalize(informaticup_data, mu=mu, sigma=sigma)

        if args.model_name == 'logistic':
            base = LogisticRegression()
            model_lr = CalibratedClassifierCV(base, cv=5)
            model_lr.fit(data, labels)

            pickle.dump(model_lr, open("models/gb_model_logistic", "wb"))     
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_lr.predict(toefl_data)
            probs = model_lr.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_lr.predict(informaticup_data)
            probs = model_lr.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
        
        if args.model_name == 'xgboost':
            xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            model_xgb = CalibratedClassifierCV(xgb_clf, cv=5)
            model_xgb.fit(data, labels)

            pickle.dump(model_xgb, open("models/gb_model_xgb", "wb"))
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_xgb.predict(toefl_data)
            probs = model_xgb.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_xgb.predict(informaticup_data)
            probs = model_xgb.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
            
        if args.model_name == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            model_rf = CalibratedClassifierCV(rf, cv=5)
            model_rf.fit(data, labels)

            pickle.dump(model_rf, open("models/gb_model_rf", "wb"))
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_rf.predict(toefl_data)
            probs = model_rf.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_rf.predict(informaticup_data)
            probs = model_rf.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
            
    if args.mle:
        mle_training_data = generate_dataset_fn(get_intrinsic_mle_file)
        mle_training_data, mu, sigma = normalize(mle_training_data, ret_mu_sigma=True)
        pickle.dump(mu, open("models/intrinsic_mu", "wb"))
        pickle.dump(sigma, open("models/intrinsic_sigma", "wb"))
        labels = generate_dataset_fn(lambda file: 1 if any([m in file for m in ["ai"]]) else 0)
        
        #Testing data
        toefl = get_generate_dataset(Dataset("normal", "data/toefl91"))
        toefl_labels = toefl(lambda _: 0)
        toefl_data = toefl(get_intrinsic_mle_file)
        toefl_data = normalize(toefl_data, mu=mu, sigma=sigma)

        informaticup = get_generate_dataset(Dataset("normal", "data/informaticup-test-dataset/informaticup-dataset/ai_gen_text"))
        informaticup_labels = informaticup(lambda _: 1)
        informaticup_data = informaticup(get_intrinsic_mle_file)
        informaticup_data = normalize(informaticup_data, mu=mu, sigma=sigma)

        if args.model_name == 'logistic':
            base = LogisticRegression()
            model_lr = CalibratedClassifierCV(base, cv=5)
            model_lr.fit(mle_training_data, labels)

            pickle.dump(model_lr, open("models/intrinsic_model_logistic", "wb"))     
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_lr.predict(toefl_data)
            probs = model_lr.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_lr.predict(informaticup_data)
            probs = model_lr.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
        
        if args.model_name == 'xgboost':
            xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            model_xgb = CalibratedClassifierCV(xgb_clf, cv=5)
            model_xgb.fit(mle_training_data, labels)

            pickle.dump(model_xgb, open("models/intrinsic_model_xgb", "wb"))
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_xgb.predict(toefl_data)
            probs = model_xgb.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_xgb.predict(informaticup_data)
            probs = model_xgb.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
            
        if args.model_name == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            model_rf = CalibratedClassifierCV(rf, cv=5)
            model_rf.fit(mle_training_data, labels)

            pickle.dump(model_rf, open("models/intrinsic_model_rf", "wb"))
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_rf.predict(toefl_data)
            probs = model_rf.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_rf.predict(informaticup_data)
            probs = model_rf.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])


    if args.combined:
        combined_training_data = generate_dataset_fn(gb_mle_combined_file)
        combined_training_data, mu, sigma = normalize(combined_training_data, ret_mu_sigma=True)
        pickle.dump(mu, open("models/comb_mu", "wb"))
        pickle.dump(sigma, open("models/comb_sigma", "wb"))
        labels = generate_dataset_fn(lambda file: 1 if any([m in file for m in ["ai"]]) else 0)

        #Testing data
        toefl = get_generate_dataset(Dataset("normal", "data/toefl91"))
        toefl_labels = toefl(lambda _: 0)
        toefl_data = toefl(gb_mle_combined_file)
        toefl_data = normalize(toefl_data, mu=mu, sigma=sigma)

        informaticup = get_generate_dataset(Dataset("normal", "data/informaticup-test-dataset/informaticup-dataset/ai_gen_text"))
        informaticup_labels = informaticup(lambda _: 1)
        informaticup_data = informaticup(gb_mle_combined_file)
        informaticup_data = normalize(informaticup_data, mu=mu, sigma=sigma)

        if args.model_name == 'logistic':
            base = LogisticRegression()
            model_lr = CalibratedClassifierCV(base, cv=5)
            model_lr.fit(combined_training_data, labels)

            pickle.dump(model_lr, open("models/comb_model_logistic", "wb"))     
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_lr.predict(toefl_data)
            probs = model_lr.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_lr.predict(informaticup_data)
            probs = model_lr.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
        
        if args.model_name == 'xgboost':
            xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            model_xgb = CalibratedClassifierCV(xgb_clf, cv=5)
            model_xgb.fit(combined_training_data, labels)

            pickle.dump(model_xgb, open("models/comb_model_xgb", "wb"))
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_xgb.predict(toefl_data)
            probs = model_xgb.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_xgb.predict(informaticup_data)
            probs = model_xgb.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])
            
        if args.model_name == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            model_rf = CalibratedClassifierCV(rf, cv=5)
            model_rf.fit(combined_training_data, labels)

            pickle.dump(model_rf, open("models/comb_model_rf", "wb"))
            print("Saved model to model/")

            #Test on Toefl
            predictions = model_rf.predict(toefl_data)
            probs = model_rf.predict_proba(toefl_data)[:, 1]

            result_table.append(
                [
                    "Toefl91",
                    round(f1_score(toefl_labels, predictions), 3),
                    round(accuracy_score(toefl_labels, predictions), 3),
                    round(roc_auc_score(toefl_labels, probs), 3),
                ])
            
            #Test on Informatic Cup
            predictions = model_rf.predict(informaticup_data)
            probs = model_rf.predict_proba(informaticup_data)[:, 1]

            result_table.append(
                [
                    "Informatic Cup",
                    round(f1_score(informaticup_labels, predictions), 3),
                    round(accuracy_score(informaticup_labels, predictions), 3),
                    round(roc_auc_score(informaticup_labels, probs), 3),
                ])

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))