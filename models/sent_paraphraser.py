from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from common import constants as const

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SentParaphraser():
    def __init__(
            self,
            data: str | List,    
            model_path: str | Path=const.PRE_TRAINED_MODELS,
            model_name: str=const.T5_PARAPHRASER,
            max_length: int=256,
            do_sample: bool=True,
            top_k: int=120,
            top_p: float=0.975,
            early_stopping: bool=True,
            num_return_sequences: int=5,
            penalty_alpha: float=0.6,
            num_beams: int=25,
        ):
        """
        The T5 paraphraser model for multiple text data inputs which generate top-N paraphrased outputs.

        params:
            data: Input text or text list for paraphrasing the input text.
            model_path: The language model where the pre-trained language model (LM) is stored.
            model_name: The name of input language model from the huggingface library.
            max_length: The maximum length of the generated tokens.
            do_sample: The use of sampling instead of greedy decoding.
            top_k: The highest probability tokens for top-k filtering.
            top_p: The smallest set of probable tokens that adds upto top_p probability.
            early_stopping: When True, it assists in exploring the beam search candidates.
            num_return_sequences: The number of returned paraphrased sequences.
            penalty_alpha: The penalty value in contrastive search decoding.
            num_beams: The number of beams for the beam search.
        """
        self.data = data
        if isinstance(self.data, str):
            self.data = [self.data]
        self.model_path = model_path
        self.model_name = model_name
        self.max_length = max_length
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.early_stopping = early_stopping
        self.num_return_sequences = num_return_sequences
        self.penalty_alpha = penalty_alpha
        self.num_beams = num_beams

        # loading the language model and its corresponding tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_path)

    def _paraphrase_text_single(self, input_text) -> Dict[str, float]:
        encoding = self.tokenizer.encode_plus(input_text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=self.max_length,
                    do_sample=self.do_sample,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    early_stopping=self.early_stopping,
                    num_return_sequences=self.num_return_sequences,
                    penalty_alpha=self.penalty_alpha,
                    num_beams=self.num_beams,
                )
        outputs_list = []              
        for output in outputs:
            outputs_list.append(str(self.tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)))
        return outputs_list

    def paraphrase_text(self):
        paraphrase_text_list = []
        for text in tqdm(self.data):
            text =  "paraphrase: " + text + " </s>"
            paraphrase_text_list.append(self._paraphrase_text_single(text))

        return paraphrase_text_list

if __name__ == '__main__':
    input_data = [
            "Frühling erwacht leise, Blumen blühen im Sonnenlicht, Natur tanzt im Wind.",
            "Jupiter, the largest planet in our solar system, has 79 known moons as of my last knowledge "
            "update in September 2021. However, new moons may have been discovered since then as space "
            "exploration and observation continue."
        ]

    sent_paraphraser = SentParaphraser(input_data)
    print(sent_paraphraser.paraphrase_text())
    del sent_paraphraser
