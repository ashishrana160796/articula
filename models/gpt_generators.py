"""
This file contains GPT2 generators for English and German languages to prepare new datasets.
"""
from typing import List
from pathlib import Path
from tqdm import tqdm

from common import constants as const
from transformers import set_seed, pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead, GPT2LMHeadModel

set_seed(16) # setting stochastic seed for generating the outputs

class GPT2Generator():
    def __init__(
            self,
            data: str | List,    
            model_path: str | Path=const.PRE_TRAINED_MODELS,
            language: str='en',
            min_length: int=16,
            max_length: int=128,
            gen_method: str='dtr'
        ):
        """
        The GPT2 generator for the input English or German datasets.

        params:
            data: Input text or text list prompt for generating from the GPT model.
            model_path: The language model path for generating the text.
            language: The language of the model used for generating texts via the prompts.
            min_length: The minimum length of the generated output.
            max_length: The maximum length of the generated output.
            gen_method: The text generation approach followed between deterministic, stochastic, and contrastive.
        """
        self.data = data
        if isinstance(self.data, str):
            self.data = [self.data]
        self.model_path = model_path
        self.language = language
        self.min_length = min_length
        self.max_length = max_length
        self.gen_method = gen_method

        if self.language == 'en':
            self.tokenizer = AutoTokenizer.from_pretrained(const.GPT_EN, cache_dir=self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(const.GPT_EN, cache_dir=self.model_path)
        elif self.language == 'de':
            self.tokenizer = AutoTokenizer.from_pretrained(const.GPT_DE, cache_dir=self.model_path)
            self.model = AutoModelWithLMHead.from_pretrained(const.GPT_DE, cache_dir=self.model_path)
        else:
            raise ValueError('The parsed language parameter is incorrect, must be from the set of values in ("de", "en").')
        
        if self.gen_method == 'dtr':
            self.pipeline = pipeline(task=const.TEX_GEN, model=self.model, tokenizer=self.tokenizer)
        elif self.gen_method == 'stc':
            self.pipeline = pipeline(task=const.TEX_GEN, model=self.model, tokenizer=self.tokenizer, top_p=0.95, top_k=0)
        elif self.gen_method == 'con':
            self.pipeline = pipeline(task=const.TEX_GEN, model=self.model, tokenizer=self.tokenizer, penalty_alpha=0.6, top_k=12)
        else:
            raise ValueError('The parsed language parameter is incorrect, must be from the set of values in ("dtr", "stc", "con").')

    def _generate_text_single(self, text_prompt):
        output_text = self.pipeline(
                                text_prompt,
                                min_length=self.min_length,
                                max_length=self.max_length,
                            )[0]["generated_text"]
        return output_text

    def generate_text(self):
        gen_text_list = []
        for text_prompt in tqdm(self.data):
            gen_text_list.append(self._generate_text_single(text_prompt))
        return gen_text_list

if __name__ == '__main__':
    input_text = 'Paraphrase the text with additional relevant information: ' \
                 '"Jupiter, the largest planet in our solar system, has 79 known moons as of my last knowledge update in September 2021."'
    generator_en = GPT2Generator(input_text, min_length=32, gen_method='con')
    print(generator_en.generate_text())
    print(generator_en.generate_text()[0].replace(input_text,""))
    del generator_en

    input_text = 'Paraphrasieren Sie den Text mit zusätzlichen relevanten Informationen: ' \
                 '"Frühling erwacht leise, Blumen blühen im Sonnenlicht, Natur tanzt im Wind."'
    generator_de = GPT2Generator(input_text, language='de', min_length=32, gen_method='con')
    print(generator_de.generate_text())
    print(generator_de.generate_text()[0].replace(input_text,""))
    del generator_de
