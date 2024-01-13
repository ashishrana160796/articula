import nltk
nltk.download('punkt')

import random
from typing import List
from pathlib import Path
from common import constants as const
from tqdm import tqdm
from copy import deepcopy

from models.sent_paraphraser import SentParaphraser
from models.gpt_generators import GPT2Generator
from models.intrinsic_dim_estimator import IntrinsicDimensionEstimator
from models.detector_models import LanguageDetector
from common.data_utils import preprocess_text, sentence_tokenizer

random.seed(16)

class BaselineTextSpoofer():
    def __init__(
        self,
        data: str | List,
        model_path: str | Path=const.PRE_TRAINED_MODELS,
        num_return_sequences: int=3,
        num_beams: int=1,
        add_info_mutation: bool=True,
        add_info_mut_prob: float=0.5,
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
        self.punkt_lang_map = {'en': 'english', 'de': 'german'}
        self.languages = [self.punkt_lang_map[lang_code] for lang_code in self.languages]
        del lang_detector


    def _spoof_text_single(self, text, language): # greedy implementation
        text_sent_list = sentence_tokenizer(text, language)
        best_spoofing_score = -1000
        mut_list = [1 if random.uniform(0, 1) <= self.add_info_mut_prob else 0 for x in range(0, len(text_sent_list))]
        sent_parahraser = SentParaphraser(data="",num_return_sequences=self.num_return_sequences, num_beams=self.num_beams)
        paraphrased_outputs = []
        for text_sent in tqdm(text_sent_list, desc="generating phrasing alternatives..."):
            sent_parahraser.data = [text_sent]
            paraphrased_outputs.append(sent_parahraser.paraphrase_text()[0])
        del sent_parahraser

        if self.add_info_mutation:
            max_length = max([len(tmp_sent.split(" ")) if mut_flg == 1 else 0 for tmp_sent, mut_flg in zip(text_sent_list, mut_list)])
            min_length = min([len(tmp_sent.split(" ")) if mut_flg == 1 else 0 for tmp_sent, mut_flg in zip(text_sent_list, mut_list)])
            gpt_language = "de" if language == "german" else "en"
            gpt_generator = GPT2Generator(
                                        data="",
                                        min_length=int(0.75 * min_length),
                                        max_length=int(3 * max_length),
                                        language=gpt_language,
                                        gen_method='con',
                                        early_stopping=True,
                                    )
            gpt_prompt = 'Paraphrase the text with additional relevant information: '
            if gpt_language == 'de':
                gpt_prompt = 'Paraphrasieren Sie den Text mit zusätzlichen relevanten Informationen: '
            for tmp_sent, para_sent_list, mut_flg in tqdm(zip(text_sent_list, paraphrased_outputs, mut_list), desc="generating additional information alternatives..."):
                if mut_flg == 1:
                    gpt_generator.data = [f'{gpt_prompt}"{tmp_sent}"']
                    tmp_gen_text = tmp_sent+" "+ preprocess_text(gpt_generator.generate_text()[0].replace(f'{gpt_prompt}"{tmp_sent}"',""))
                    para_sent_list.append(tmp_gen_text)
            del gpt_generator

        dim_estimator = IntrinsicDimensionEstimator("")
        for para_sents, idx in tqdm(zip(paraphrased_outputs, list(range(len(paraphrased_outputs)))), desc="selecting best phrasing alternatives..."):
            for para_sent in para_sents:
                tmp_text_sent_list = deepcopy(text_sent_list)
                tmp_text_sent_list[idx] = para_sent
                tmp_spoof_text = " ".join(tmp_text_sent_list)
                dim_estimator.data = [tmp_spoof_text]
                tmp_spoofing_score = dim_estimator.get_mle().ravel()[0]
                if tmp_spoofing_score > best_spoofing_score:
                    best_spoofing_score = tmp_spoofing_score
                    text_sent_list[idx] = para_sent

        return " ".join(text_sent_list), best_spoofing_score


    def spoof_text(self):
        spoof_text_list = []
        for text, text_lang in zip(self.data, self.languages):
            spoof_text_list.append(self._spoof_text_single(preprocess_text(text), text_lang))
        return spoof_text_list


if __name__ == '__main__':
    # input_data = [
    #         "Berlin, die Hauptstadt Deutschlands, ist eine dynamische Stadt, die sowohl eine reiche Geschichte "
    #         "als auch eine lebendige zeitgenössische Kultur verkörpert. Ihr ikonisches Brandenburger Tor steht "
    #         "als Symbol für Einheit und Versöhnung, während die Überreste der Berliner Mauer eine eindringliche "
    #         "Erinnerung an die geteilte Vergangenheit der Stadt darstellen.",
    #         "Jupiter, the largest planet in our solar system, has 79 known moons as of my last knowledge "
    #         "update in September 2021. However, new moons may have been discovered since then as space "
    #         "exploration and observation continue."
    #     ]
    # print(sentence_tokenizer(input_data[0], language='german'))
    # print(sentence_tokenizer(input_data[1], language='english'))
    input_text = ["Sehr geehrte Damen und Herren, "
            ""
            "mit großem Interesse bewerbe ich mich für die Stelle als Fachangestellter im Bürohandel in Ihrem "
            "Unternehmen. Aufgrund meiner fundierten Erfahrung im Bürohandel und meiner Begeisterung für die "
            "Organisation und Verwaltung von Büromaterialien, sehe ich diese Position als eine hervorragende Gelegenheit, "
            "meine Fähigkeiten und Fachkenntnisse einzubringen."
            ""
            "Während meiner bisherigen beruflichen Laufbahn habe ich umfangreiche Kenntnisse im Einkauf und in der "
            "Lagerverwaltung von Bürobedarf erworben. Ich bin vertraut mit verschiedenen Bürosoftwareanwendungen und "
            "habe eine ausgezeichnete Fähigkeit zur Kundenbetreuung entwickelt. Zudem bin ich äußerst organisiert, "
            "detailorientiert und arbeite effizient, um sicherzustellen, dass die Büroprodukte stets verfügbar "
            "sind und den Bedürfnissen der Kunden gerecht werden."
            ""
            "Meine Leidenschaft für den Bürohandel und mein Engagement für exzellenten Kundenservice haben mich dazu "
            "motiviert, stets auf dem neuesten Stand der Entwicklungen in der Branche zu bleiben. Ich bin überzeugt, "
            "dass ich eine wertvolle Ergänzung für Ihr Team sein kann und freue mich darauf, dazu beizutragen, "
            "die hohen Standards Ihres Unternehmens aufrechtzuerhalten."
            ""
            "Vielen Dank für Ihre Zeit und die Berücksichtigung meiner Bewerbung. Ich stehe Ihnen gerne für ein "
            "persönliches Gespräch zur Verfügung, um meine Qualifikationen und Motivation weiter zu erläutern. "
            ""
            "Mit freundlichen Grüßen,"
            "[Ihr Name]",]
    text_spoofer = BaselineTextSpoofer(input_text, add_info_mutation=True)
    print(text_spoofer.spoof_text())
