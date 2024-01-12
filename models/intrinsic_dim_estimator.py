import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel

from scipy.spatial.distance import cdist
from skdim.id import MLE

from tqdm import tqdm
from typing import Callable, List

from common.data_utils import preprocess_text
from common.intrinsic_dim_utils import PHD
from common import constants as const
from common.data_utils import load_informaticup_text_data
from models.detector_models import LanguageDetector

class IntrinsicDimensionEstimator():
    def __init__(
            self,
            data: str | List,
            estimator_type: str='PHD',
            model_path: str | Path=const.PRE_TRAINED_MODELS,
            model_name: str=const.ROBERTA_BASE,
            min_subsample: int=40,
            intermediate_points: int=7,
            alpha: float=1.0,
            metric: str | Callable='euclidean',
            n_reruns: int=3,
            n_points: int=9,
            n_points_min: int=3,
        ):
        """
        The intrinsic dimension estimator for the input text string or list using PHD or MLE metrics.

        params:
            data: Input text or text list for calculating the intrinsic dimension.
            model_path: The language model where the pre-trained language model (LM) is stored.
            model_name: The name of input language model from the huggingface library.
            min_subsample: The size of the minimal subsample to be drawn in procedure, and its less value gives statisitcally stable predictions.
            intermediate_points: The number of subsamples to be drawn, and its higher value gives stable dimension estimation.
            alpha: A real-valued parameter "alpha" for computing PH-dim. The "alpha" should be chosen lower than 
                the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
            metric: A distance function for the metric space (see documentation for scipy.spatial.distance.cdist).
            n_reruns: The number of restarts of whole calculations (each restart is made in a separate thread).
            n_points: The number of subsamples to be drawn at each subsample.
            n_points_min: The number of subsamples to be drawn at larger subsamples (more than half of the point cloud).
        """
        self.data = data
        if isinstance(self.data, str):
            self.data = [self.data]
        self.model_path = model_path
        self.model_name = model_name
        self.min_subsample = min_subsample
        self.intermediate_points = intermediate_points
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
    
        # loading the language model and its corresponding tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name, cache_dir=self.model_path)
        self.model = RobertaModel.from_pretrained(self.model_name, cache_dir=self.model_path)
        self.lang_detector = LanguageDetector(self.data)
        self.lang_lists = self.lang_detector.detect_language()

    def _get_phd_single(self, text, solver):
        inputs = self.tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outp = self.model(**inputs)
    
        # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
        mx_points = inputs['input_ids'].shape[1] - 2

        mn_points = self.min_subsample
        step = ( mx_points - mn_points ) // self.intermediate_points
        
        return solver.fit_transform(outp[0][0].numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, \
                                    point_jump=step)
    
    def _get_phd_single_multi_ling(self, text, solver):
        inputs = self.lang_detector.tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outp = self.lang_detector.model(**inputs)
    
        # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
        mx_points = inputs['input_ids'].shape[1] - 2

        mn_points = self.min_subsample
        step = ( mx_points - mn_points ) // self.intermediate_points
        
        return solver.fit_transform(outp[0][0].numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, \
                                    point_jump=step)

    def get_phd(self):
        dims = []
        phd_solver = PHD(alpha=self.alpha, metric=self.metric, n_points=self.n_points)
        for text, lang in tqdm(zip(self.data, self.lang_lists)):
            if lang == 'en':
                dims.append(self._get_phd_single(text, phd_solver))
            else:
                dims.append(self._get_phd_single_multi_ling(text, phd_solver))

        return np.array(dims).reshape(-1, 1)

    def _get_mle_single(self, text, solver):
        inputs = self.tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outp = self.model(**inputs)

        return solver.fit_transform(outp[0][0].numpy()[1:-1])

    def _get_mle_single_multi_ling(self, text, solver):
        inputs = self.lang_detector.tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outp = self.lang_detector.model(**inputs)

        return solver.fit_transform(outp[0][0].numpy()[1:-1])

    def get_mle(self):
        dims = []
        MLE_solver = MLE()
        for text, lang in tqdm(zip(self.data, self.lang_lists)):
            if lang == 'en':
                dims.append(self._get_mle_single(text, MLE_solver))
            else:
                dims.append(self._get_mle_single_multi_ling(text, MLE_solver))

        return np.array(dims).reshape(-1, 1)
    
if __name__ == '__main__':
    # evaluating intrinsic dimensions over sample texts
    input_texts = [
            "Speaking of festivities, there is one day in China that stands unrivaled"
            "the first day of the Lunar New Year, commonly referred to as the Spring Festival"
            "Even if you're generally uninterested in celebratory events, it's hard to resist the"
            "allure of the family reunion dinner, a quintessential aspect of the Spring Festival."
            "Throughout the meal, family members raise their glasses to toast one another, expressing wishes"
            "for happiness, peace, health, and prosperity in the upcoming year.",
            "Berlin, die Hauptstadt Deutschlands, ist eine dynamische Stadt, die sowohl eine reiche Geschichte "
            "als auch eine lebendige zeitgenössische Kultur verkörpert. Ihr ikonisches Brandenburger Tor steht "
            "als Symbol für Einheit und Versöhnung, während die Überreste der Berliner Mauer eine eindringliche"
            "Erinnerung an die geteilte Vergangenheit der Stadt darstellen. Mit einer blühenden Kunstszene, "
            "vielfältigen kulinarischen Angeboten und einer jugendlichen Energie ist Berlin eine Stadt, die "
            "sich ständig neu erfindet und somit ein Muss für Reisende ist, die eine Mischung aus Tradition und Innovation suchen.",
        ]
    
    text_dim_estimator = IntrinsicDimensionEstimator(input_texts)
    print(text_dim_estimator.lang_lists)
    print(text_dim_estimator.get_mle())
    print(text_dim_estimator.get_phd())

    input_data = load_informaticup_text_data()
    text_dim_estimator = IntrinsicDimensionEstimator(input_data)
    print(text_dim_estimator.lang_lists)
    print(text_dim_estimator.get_mle())
