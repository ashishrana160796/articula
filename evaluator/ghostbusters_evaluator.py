import numpy as np
import dill as pickle
import tiktoken
import os
import argparse

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from sklearn.linear_model import LogisticRegression
from common.ghostbusters_featurize import normalize, t_featurize_logprobs, score_ngram
from common.ghostbusters_symbolic import train_trigram, get_words, vec_functions, scalar_functions

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="input.txt")
args = parser.parse_args()

file = args.file
MAX_TOKENS = 512
best_features = open("models/gb_features_all.txt").read().strip().split("\n")

# Load davinci tokenizer
enc = tiktoken.encoding_for_model("davinci")

# Load model
model = pickle.load(open("models/gb_model_logistic", "rb"))
mu = pickle.load(open("models/gb_mu", "rb"))
sigma = pickle.load(open("models/gb_sigma", "rb"))

# Load data and featurize
with open(file) as f:
    doc = f.read().strip()
    # Strip data to first MAX_TOKENS tokens
    tokens = enc.encode(doc)[:MAX_TOKENS]
    doc = enc.decode(tokens).strip()

    print(f"Input: {doc}")

# Load/Train trigram
trigram_model_path = "models/trigram_model_mling.pkl"
if not os.path.exists(trigram_model_path):
    print("Training trigram model...")
    trigram_model = train_trigram(verbose=True, return_tokenizer=False)
else:
    print("Loading trigram model...")
    trigram_model = pickle.load(
        open(trigram_model_path, "rb"), pickle.HIGHEST_PROTOCOL)

trigram = np.array(score_ngram(doc, trigram_model, enc.encode, n=3, strip_first=False))
unigram = np.array(score_ngram(doc, trigram_model.base, enc.encode, n=1, strip_first=False))

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

tokens = tokenizer_gpt2(doc, return_tensors="pt", max_length=MAX_TOKENS, truncation=True)
    
with torch.no_grad():
    outputs_gpt2 = model_gpt2(**tokens, labels=tokens["input_ids"])
    
logits_gpt2 = outputs_gpt2.logits
log_probs_gpt2 = torch.nn.functional.log_softmax(logits_gpt2, dim=-1)

# Load DistilBERT Multilingual model
db_model_name = "distilbert-base-multilingual-cased"
db_model = AutoModelForMaskedLM.from_pretrained(db_model_name)

# Get model output
db_model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs_db = db_model(**tokens, labels=tokens["input_ids"])
    
# Calculate log probabilities
logits_db = outputs_db.logits
log_probs_db = torch.nn.functional.log_softmax(logits_db, dim=-1)

# Extract log probabilities for each token
gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
token_log_probs_gpt2 = []
token_log_probs_db = []
subwords = []
for i, input_id in enumerate(tokens["input_ids"][0]):
    token_log_prob_gpt2 = log_probs_gpt2[0, i, input_id].item()
    token_log_prob_db = log_probs_db[0, i, input_id].item()
    token_text = tokenizer_gpt2.decode([input_id])
    for k, v in gpt2_map.items():
        token_text = token_text.replace(k, v)
        
    subwords.append(token_text)
    token_log_probs_gpt2.append(token_log_prob_gpt2)
    token_log_probs_db.append(token_log_prob_db)

token_log_probs_gpt2 = np.array(token_log_probs_gpt2)
token_log_probs_db = np.array(token_log_probs_db)

t_features = t_featurize_logprobs(token_log_probs_gpt2, token_log_probs_db, subwords)

vector_map = {
    "gpt2-logprobs": np.array(token_log_probs_gpt2),
    "db-logprobs": np.array(token_log_probs_db),
    "trigram-logprobs": trigram,
    "unigram-logprobs": unigram
}

exp_features = []
for exp in best_features:

    exp_tokens = get_words(exp)
    curr = vector_map[exp_tokens[0]]

    for i in range(1, len(exp_tokens)):
        if exp_tokens[i] in vec_functions:
            next_vec = vector_map[exp_tokens[i+1]]
            curr = vec_functions[exp_tokens[i]](curr, next_vec)
        elif exp_tokens[i] in scalar_functions:
            exp_features.append(scalar_functions[exp_tokens[i]](curr))
            break

data = (np.array(t_features + exp_features) - mu) / sigma
preds = model.predict_proba(data.reshape(-1, 1).T)[:, 1]

print(f"Prediction: {preds}")
