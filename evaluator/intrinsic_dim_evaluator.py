import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

import xgboost as xgb
from omegaconf import OmegaConf
from scipy.stats import sem
from numpy import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import constants as const
from common.data_utils import intrinsic_dim_dataset_to_csv, load_informaticup_text_data
from models.intrinsic_dim_estimator import IntrinsicDimensionEstimator
from common.language_model_utils import GenreDetector


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=const.INTRINSIC_DIM_DATASET)
parser.add_argument('--save_path', type=str, default=const.TRANSFORMED_DATA_SAVE_PATH)
parser.add_argument('--file_name', type=str, default=const.TRANSFORMED_DATA_CSV_NAME)
parser.add_argument('--dataset_type', type=str, choices=['intrinsic', 'informaticup'], default='informaticup')
parser.add_argument('--get_intrinsic_dim_estimates', type=bool, default=False)
parser.add_argument('--subset_fraction', type=int, default=0.125)
parser.add_argument('--append_prompt', type=bool, default=True)
parser.add_argument('--model_name', type=str, choices=['logistic', 'xgboost'], default='logistic')
parser.add_argument('--conf_path', type=str, default=const.XGBOOST_CONF_PATH)
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

def load_informaticup_dataset(input_features: List[str]= ['mle_value', 'genre_value']):
    infcp_txt_data = load_informaticup_text_data()
    text_dim_estimator = IntrinsicDimensionEstimator(infcp_txt_data)
    mle_values = text_dim_estimator.get_mle()
    mle_values = [mle_num for mle_value in mle_values for mle_num in mle_value]
    del text_dim_estimator
    genre_detector = GenreDetector(infcp_txt_data)
    genre_values = genre_detector.detect_genre()
    del genre_detector
    y = np.zeros(len(mle_values))
    if len(input_features) == 1:
        X = np.array(mle_values).reshape(-1, 1)
        return X, y

    X = np.array([mle_values, genre_values]).T
    return X, y


def logistic_model_evaluator(X: np.ndarray, y: np.ndarray, num_repeats: int, n_splits: int=5):
    cv_splits = RepeatedKFold(n_splits=n_splits, n_repeats=num_repeats, random_state=1)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv_splits, n_jobs=-1)
    return scores


def get_trained_logistic_model(X: np.ndarray, y: np.ndarray):
    model = LogisticRegression(random_state=16)
    model.fit(X, y)
    return model


def xgb_model_evaluator(X: np.ndarray, y: np.ndarray, params_conf_path: str, n_splits: int=5, metics: str='error'):
    data_dmatrix = xgb.DMatrix(data=X,label=y)
    xgb_params = dict(OmegaConf.load(params_conf_path))
    xgb_cv_df = xgb.cv(dtrain=data_dmatrix, params=xgb_params, nfold=n_splits, metrics=metics, as_pandas=True, seed=16)
    return xgb_cv_df

if __name__ == '__main__':
    if args.get_intrinsic_dim_estimates:
        calculate_intrinsic_dim_estimates(
                args.data_path,
                args.save_path,
                args.file_name,
                args.subset_fraction,
                args.append_prompt,
            )

    input_features_list = [['phd_value'], ['mle_value'], ['mle_value', 'phd_value']]
    if args.dataset_type == 'informaticup':
        input_features_list = [['mle_value', 'text_genre'], ['mle_value']]
    if args.dataset_type != 'informaticup':
        if args.model_name == 'logistic':
            for input_features in input_features_list:
                X, y = load_dataset(args.save_path, args.file_name, input_features)
                scores = logistic_model_evaluator(X, y, args.num_repeats, args.num_k_folds)
                print(f'Input Features: {input_features} yields mean value of {mean(scores)} with standard error {sem(scores)}')
        elif args.model_name == 'xgboost':
            for input_features in input_features_list:
                X, y = load_dataset(args.save_path, args.file_name, input_features)
                scores_df = xgb_model_evaluator(X, y, args.conf_path, args.num_k_folds) 
                print(f"Input Features: {input_features} yields mean value of {(1 - mean(np.array(scores_df['train-error-mean'])))*100}"
                    f"with standard error {mean(np.array(scores_df['train-error-std']))*100} for the train set.")
                print(f"Input Features: {input_features} yields mean value of {(1 - mean(np.array(scores_df['test-error-mean'])))*100}"
                    f"with standard error {mean(np.array(scores_df['test-error-std']))*100} for the test set.")
    else:
        if args.model_name == 'logistic':
            for input_features in input_features_list:
                X_test, y_test = load_informaticup_dataset(input_features)
                X, y = load_dataset(args.save_path, args.file_name, input_features)
                log_reg_model = get_trained_logistic_model(X, y)
                y_pred = log_reg_model.predict(X_test)
                print(f'Input Features: {input_features} yields accuracy of {accuracy_score(y_test, y_pred)}')

