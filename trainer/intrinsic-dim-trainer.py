import pandas as pd
from pathlib import Path
from typing import Optional

from common import constants as const
from common.data_utils import intrinsic_dim_dataset_to_csv
from models.intrinsic_dim_estimator import IntrinsicDimensionEstimator

def calculate_intrinsic_dim_estimates(data_path: str | Path, save_path: str | Path, file_name: str | Path, subset_fraction: Optional[float]=1.):
    intrinsic_dim_dataset_to_csv(data_path, subset_fraction)
    intrinsic_dim_df = pd.read_csv(Path(save_path) / file_name)
    text_dim_estimator = IntrinsicDimensionEstimator(intrinsic_dim_df['text_data'])
    text_dim_mle = text_dim_estimator.get_mle()
    text_dim_phd = text_dim_estimator.get_phd()
    intrinsic_dim_df["mle_value"] = text_dim_mle
    intrinsic_dim_df["phd_value"] = text_dim_phd
    intrinsic_dim_df = intrinsic_dim_df[['text_data', 'generator', 'data_source', 'data_split', 'mle_value', 'phd_value', 'target_class']]
    intrinsic_dim_df.to_csv(Path(save_path) / file_name, index=False)

if __name__ == '__main__':
    calculate_intrinsic_dim_estimates(const.INTRINSIC_DIM_DATASET, const.TRANSFORMED_DATA_SAVE_PATH, const.TRANSFORMED_DATA_CSV_NAME, 0.125)
