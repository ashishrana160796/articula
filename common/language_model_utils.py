from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from common import constants as const
from common.data_utils import load_informaticup_text_data

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

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
        self.model_classification = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_path)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.model_path)

    def _detect_language_single(self, input_text) -> Dict[str, float]:
        inputs = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model_classification(**inputs).logits

        preds = torch.softmax(logits, dim=-1)

        # Map raw predictions to languages
        id2lang = self.model_classification.config.id2label
        vals, idxs = torch.max(preds, dim=1)
        lang_pred_dict = {id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)}
        lang_pred_dict_values = list(lang_pred_dict.values())
        lang_pred_dict_keys = list(lang_pred_dict.keys())
        return lang_pred_dict_keys[lang_pred_dict_values.index(max(lang_pred_dict_values))]

    def detect_language(self):
        detected_lang_list = []
        for text in tqdm(self.data):
            detected_lang_list.append(self._detect_language_single(text))

        return detected_lang_list

if __name__ == '__main__':
    input_data = [
            "Frühling erwacht leise, Blumen blühen im Sonnenlicht, Natur tanzt im Wind.",
            "Jupiter, the largest planet in our solar system, has 79 known moons as of my last knowledge "
            "update in September 2021. However, new moons may have been discovered since then as space "
            "exploration and observation continue.",
            "Brevity is the soul of wit.",
            "Amor, ch'a nullo amato amar perdona.",
        ]

    lang_detector = LanguageDetector(input_data)
    print(lang_detector.detect_language())

    input_data = load_informaticup_text_data()
    lang_detector = LanguageDetector(input_data)
    print(lang_detector.detect_language())
