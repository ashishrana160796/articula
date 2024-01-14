import json
import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
else:
    print("Using CPU...")
    device = torch.device("cpu")

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Load DistilBERT Multilingual model and tokenizer
db_model_name = "distilbert-base-multilingual-cased"
db_model = AutoModelForMaskedLM.from_pretrained(db_model_name).to(device)

def write_logprobs_gpt2(text, file):
    """
    Run text under gpt2 model and write logprobs to file, separated by newline.
    """

    max_length = model_gpt2.config.n_positions
    tokens = tokenizer_gpt2(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model_gpt2(**tokens, labels=tokens["input_ids"])
    
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Extract log probabilities for each token
    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
    token_log_probs = []
    for i, input_id in enumerate(tokens["input_ids"][0]):
        token_log_prob = log_probs[0, i, input_id].item()
        token_text = tokenizer_gpt2.decode([input_id])
        for k, v in gpt2_map.items():
            token_text = token_text.replace(k, v)

        token_log_probs.append((token_text, token_log_prob))

    # Write results to a text file
    with open(file, "w") as f:
        for token, log_prob in token_log_probs:
            f.write(f"{token} {-log_prob}\n")

def write_logprobs_db(text, file):
    """
    Run text under distilbert model and write logprobs to file, separated by newline.
    """
    max_length = db_model.config.max_position_embeddings
    tokens = tokenizer_gpt2(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

    # Get model output
    db_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = db_model(**tokens, labels=tokens["input_ids"])
    
    # Calculate log probabilities
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Extract log probabilities for each token
    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
    token_log_probs = []
    for i, input_id in enumerate(tokens["input_ids"][0]):
        token_log_prob = log_probs[0, i, input_id].item()
        token_text = tokenizer_gpt2.decode([input_id])
        for k, v in gpt2_map.items():
            token_text = token_text.replace(k, v)

        token_log_probs.append((token_text, token_log_prob))

    # Write results to a text file
    with open(file, "w") as f:
        for token, log_prob in token_log_probs:
            f.write(f"{token} {-log_prob}\n")