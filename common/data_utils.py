import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import OrderedDict
from common import constants as const

random.seed(16) # fixing the random seed for retrieving fixed data subset

def intrinsic_dim_dataset_to_csv(data_path: str | Path, subset_fraction: Optional[float]=1.):
    """
    It loads data subset for the intrinsic dimension dataset into a single csv file,
    with additional feature to load fraction of data subsets.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    dataset_file_paths = [entity for entity in data_path.glob('**/*') if entity.is_file()]
    
    human_text_data_values, human_text_split_values = [], []
    ai_gen_text_data_values, ai_gen_text_split_values = [], []
    for dataset_file_path in tqdm(dataset_file_paths):
        with open(dataset_file_path) as json_data_file:
            json_list = json.load(json_data_file)
            json_list = random.sample(json_list, int(subset_fraction*len(json_list)))
            for json_value in json_list:
                human_text_data_values.append(json_value['prefix']+" "+json_value['gold_completion'])
                human_text_split_values.append(json_value['split'])
                ai_gen_text_data_values.append(json_value['prefix']+" "+ ''.join(json_value['gen_completion']))
                ai_gen_text_split_values.append(json_value['split'])

    human_text_flag_values = [const.HUMAN_TEXT_STR]*len(human_text_data_values)
    ai_gen_text_flag_values = [const.AI_GENERATED_TEXT_STR]*len(ai_gen_text_data_values)

    human_text_data_df = pd.DataFrame(
        {
            
            'text_data' : human_text_data_values,
            'data_split' : human_text_split_values,
            'target_class': human_text_flag_values,
        })
    ai_gen_text_data_df = pd.DataFrame(
        {
            
            'text_data' : ai_gen_text_data_values,
            'data_split' : ai_gen_text_split_values,
            'target_class': ai_gen_text_flag_values,
        })
    
    trasformed_data_df = pd.concat([human_text_data_df, ai_gen_text_data_df], axis=0)
    trasformed_data_df.to_csv(Path(const.TRANSFORMED_DATA_SAVE_PATH) / "transformed_intrinsic_dim_data.csv", index=False)


def intrinsic_dim_dataset_to_files(data_path: str | Path, subset_fraction: Optional[float]=1.):
    """
    It loads data subset for the intrinsic dimension dataset into text files with data subset folders,
    with additional feature to load fraction of data subsets.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    dataset_file_paths = [entity for entity in data_path.glob('**/*') if entity.is_file()]

    dataset_file_paths_dict = OrderedDict()
    for file in dataset_file_paths:
        dataset_file_paths_dict[file] = [Path(file_ext) for file_ext in file.name.split(".")[0].split("_")]
    
    for (dataset_file_path, dataset_ext_list) in tqdm(dataset_file_paths_dict.items()):

        # intrinsic dimension dataset creation in GHOSTBUSTERS format
        intrinsic_dim_data = Path(const.INTRINSIC_DIM_DATA_STR)
        human_data_path = const.TRANSFORMED_DATA_SAVE_PATH / intrinsic_dim_data / dataset_ext_list[-1] / Path("human")
        ai_data_path = const.TRANSFORMED_DATA_SAVE_PATH / intrinsic_dim_data / dataset_ext_list[-1] / Path("ai")
        human_data_path.mkdir(parents=True, exist_ok=True)
        ai_data_path.mkdir(parents=True, exist_ok=True)

        with open(dataset_file_path) as json_data_file:
            json_list = json.load(json_data_file)
            IDX_VAL = 1
            for json_value in json_list:
                with open(human_data_path / f"{dataset_ext_list[0]}_{IDX_VAL}_{dataset_ext_list[1]}", "w") as f:
                    f.write(json_value['prefix']+" "+json_value['gold_completion'])
                with open(ai_data_path / f"{dataset_ext_list[1]}_{IDX_VAL}", "w") as f:
                    f.write(json_value['prefix']+" "+ ''.join(json_value['gen_completion']))
                IDX_VAL += 1

if __name__ == '__main__':
    # intrinsic_dim_dataset_to_csv(const.INTRINSIC_DIM_DATASET, subset_fraction=1.)
    intrinsic_dim_dataset_to_files(const.INTRINSIC_DIM_DATASET)
