from typing import Dict, List
from pathlib import Path
from common import constants as const

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LanguageDetector():
    def __init__(
            self,
            data: str | List,    
            model_path: str | Path=const.PRE_TRAINED_MODELS,
            model_name: str=const.XLM_ROBERTA_BASE,
        ):
        """
        The intrinsic dimension estimator for the input text string or list using PHD or MLE metrics.

        params:
            data: Input text or text list for calculating the intrinsic dimension.
            model_path: The language model where the pre-trained language model (LM) is stored.
            model_name: The name of input language model from the huggingface library.
        """
        self.data = data
        if isinstance(self.data, str):
            self.data = [self.data]
        self.model_path = model_path
        self.model_name = model_name

        # loading the language model and its corresponding tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_path)

    def detect_language(self) -> Dict[str, float]:
        inputs = self.tokenizer(self.data, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        preds = torch.softmax(logits, dim=-1)

        # Map raw predictions to languages
        id2lang = self.model.config.id2label
        vals, idxs = torch.max(preds, dim=1)
        return {id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)}


if __name__ == '__main__':
    input_data = [
            "Frühling erwacht leise, Blumen blühen im Sonnenlicht, Natur tanzt im Wind.",
            "Jupiter, the largest planet in our solar system, has 79 known moons as of my last knowledge \
            update in September 2021. However, new moons may have been discovered since then as space \
            exploration and observation continue.",
            "Brevity is the soul of wit.",
            "Amor, ch'a nullo amato amar perdona.",
        ]

    lang_detector = LanguageDetector(input_data)
    print(lang_detector.detect_language())
