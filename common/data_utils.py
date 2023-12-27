import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from common import constants as const

random.seed(16) # fixing the random seed for retrieving fixed data subset

def load_intrinsic_dim_dataset(data_path: str | Path, subset_fraction: Optional[float]=1.):
    """
    It loads data subset for the intrinsic dimension dataset,
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

if __name__ == '__main__':
    sample_text = "Speaking of festivities, there is one day in China that stands unrivaled - \
                   the first day of the Lunar New Year, commonly referred to as the Spring Festival. \
                   Even if you're generally uninterested in celebratory events, it's hard to resist the \
                   allure of the family reunion dinner, a quintessential aspect of the Spring Festival. \
                   Throughout the meal, family members raise their glasses to toast one another, expressing wishes \
                   for happiness, peace, health, and prosperity in the upcoming year."
    load_intrinsic_dim_dataset(const.INTRINSIC_DIM_DATASET, subset_fraction=.025)
