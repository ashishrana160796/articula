import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from typing import List
from pathlib import Path
from common import constants as const
from tqdm import tqdm

from models.sent_paraphraser import SentParaphraser
from models.gpt_generators import GPT2Generator
from models.intrinsic_dim_estimator import IntrinsicDimensionEstimator
from models.detector_models import LanguageDetector


def sentence_tokenizer(text, language):
    x = sent_tokenize(text, language=language)
    return x


class BaselineTextSpoofer():
    def __init__(
        self,
        data: str | List,
        model_path: str | Path=const.PRE_TRAINED_MODELS,
        num_return_sequences: int=4,
        num_beams: int=1,
        add_info_mutation: bool=True,
        add_info_mut_prob: float=0.25,
    ):
        """
        The T5 paraphraser model is used to change individual sentences iteratively and parallely obtaining feedback via the text detectors.
        And, we also introduce additional information mutation possibility to introduce extra topic information.

        params:
            data: Input text or text list for paraphrasing the input text.
            model_path: The language model where the pre-trained language model (LM) is stored.
            population_size: The number of initial population candidates.
            mul_factor: The population reduction factor after each GA round execution.
            num_return_sequences: The number of returned paraphrased sequences.
            num_beams: The number of beams for the beam search.
            add_info_mutation: The additional information inclusion during the paraphrasing stage.
            add_info_mut_prob: The mutation probability amongst the selected paraphrased candidates.
        """
        self.data = data
        if isinstance(self.data, str):
            self.data = [self.data]
        self.model_path = model_path
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams
        self.add_info_mutation = add_info_mutation
        self.add_info_mut_prob = add_info_mut_prob
        
        # detect the language, and delete the detector instance
        lang_detector = LanguageDetector(self.data)
        self.languages = lang_detector.detect_language()
        del lang_detector


    def _spoof_text_single(text, language):
        text_sent_list = sentence_tokenizer(text, language)
        

    def spoof_text(self):
        spoof_text_list = []
        for text, text_lang in tqdm(self.data, self.languages):
            spoof_text_list.append(self._spoof_text_single(text, text_lang))
        return spoof_text_list


if __name__ == '__main__':
    input_data = [
            "Berlin, die Hauptstadt Deutschlands, ist eine dynamische Stadt, die sowohl eine reiche Geschichte "
            "als auch eine lebendige zeitgenössische Kultur verkörpert. Ihr ikonisches Brandenburger Tor steht "
            "als Symbol für Einheit und Versöhnung, während die Überreste der Berliner Mauer eine eindringliche "
            "Erinnerung an die geteilte Vergangenheit der Stadt darstellen.",
            "Jupiter, the largest planet in our solar system, has 79 known moons as of my last knowledge "
            "update in September 2021. However, new moons may have been discovered since then as space "
            "exploration and observation continue."
        ]
    print(sentence_tokenizer(input_data[0], language='german'))
    print(sentence_tokenizer(input_data[1], language='english'))
