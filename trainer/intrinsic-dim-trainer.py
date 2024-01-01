import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

from scipy.stats import sem
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from common import constants as const
from common.data_utils import intrinsic_dim_dataset_to_csv
from models.intrinsic_dim_estimator import IntrinsicDimensionEstimator

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=const.INTRINSIC_DIM_DATASET)
parser.add_argument('--save_path', type=str, default=const.TRANSFORMED_DATA_SAVE_PATH)
parser.add_argument('--file_name', type=str, default=const.TRANSFORMED_DATA_CSV_NAME)
parser.add_argument('--get_intrinsic_dim_estimates', type=bool, default=False)
parser.add_argument('--subset_fraction', type=int, default=0.125)
parser.add_argument('--append_prompt', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='logistic')
parser.add_argument('--num_repeats', type=int, default=5)
parser.add_argument('--num_k_folds', type=int, default=5)
args = parser.parse_args()


def calculate_intrinsic_dim_estimates(data_path: str | Path, save_path: str | Path, file_name: str | Path, subset_fraction: Optional[float]=1., include_prompt: bool=False):
    intrinsic_dim_dataset_to_csv(data_path, subset_fraction, include_prompt)
    intrinsic_dim_df = pd.read_csv(Path(save_path) / file_name)
    text_dim_estimator = IntrinsicDimensionEstimator(intrinsic_dim_df['text_data'])
    text_dim_mle = text_dim_estimator.get_mle()
    text_dim_phd = text_dim_estimator.get_phd()
    intrinsic_dim_df["mle_value"] = text_dim_mle
    intrinsic_dim_df["phd_value"] = text_dim_phd
    intrinsic_dim_df = intrinsic_dim_df[['text_data', 'generator', 'data_source', 'data_split', 'mle_value', 'phd_value', 'target_class']]
    intrinsic_dim_df.to_csv(Path(save_path) / file_name, index=False)


def load_dataset(save_path: str | Path, file_name: str | Path, input_features: List[str]= ['mle_value', 'phd_value']):
    intrinsic_dim_df = pd.read_csv(Path(save_path) / file_name)
    intrinsic_dim_df.loc[intrinsic_dim_df["target_class"] == "human_text", "target_class"] = 1.
    intrinsic_dim_df.loc[intrinsic_dim_df["target_class"] == "ai_gen_text", "target_class"] = 0.
    intrinsic_dim_df["target_class"] = pd.to_numeric(intrinsic_dim_df["target_class"])
    y = np.array(intrinsic_dim_df["target_class"])
    
    if len(input_features) == 1:
        if input_features[0] == 'mle_value':
            X = np.array(intrinsic_dim_df["mle_value"]).reshape(-1, 1)
            return X, y
        elif input_features[0] == 'phd_value':
            X = np.array(intrinsic_dim_df["phd_value"]).reshape(-1, 1)
            return X, y
    
    X = np.array([intrinsic_dim_df["mle_value"], intrinsic_dim_df["phd_value"]]).T
    return X, y


def logistic_model_evaluator(X: np.ndarray, y: np.ndarray, num_repeats: int, n_splits: int):
    cv_splits = RepeatedKFold(n_splits=10, n_repeats=num_repeats, random_state=1)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv_splits, n_jobs=-1)
    return scores


if __name__ == '__main__':
    if args.get_intrinsic_dim_estimates:
        calculate_intrinsic_dim_estimates(
                args.data_path,
                args.save_path,
                args.file_name,
                args.subset_fraction,
                args.append_prompt,
            )

    if args.model_name == 'logistic':
        input_features_list = [['phd_value'], ['mle_value'], ['mle_value', 'phd_value']]
        for input_features in input_features_list:
            X, y = load_dataset(args.save_path, args.file_name, input_features)
            scores = logistic_model_evaluator(X, y, args.num_repeats, args.num_k_folds)
            print(f'Input Features: {input_features} yields mean value of {mean(scores)} with standard error {sem(scores)}')
