import re
import json
import random
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import zipfile
from collections import OrderedDict
from common import constants as const

random.seed(16) # fixing the random seed for retrieving fixed data subset

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def sentence_tokenizer(text, language):
    x = sent_tokenize(text, language=language)
    return x

def extract_zip_to_folder(zip_file_path):
    # Check if the file path is a zip file
    if not zip_file_path.endswith('.zip'):
        raise ValueError("The file is not a zip archive")

    # Extract the folder name (without the .zip extension)
    folder_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    folder_path = os.path.join(os.path.dirname(zip_file_path), folder_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    print(f"Extracted to {folder_path}")


def preprocess_text(text: str):
    """
    It clear text from linebreaks and odd whitespaces, because they seem to interfere with the LM.
    As a possible improvement, feel free to replace it with a more sophisticated cleaner.
    """
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    return _RE_COMBINE_WHITESPACE.sub(" ", text.replace('\n', ' ')).strip() 


def load_informaticup_text_data():
    data_list = []
    file_path_list = [file for file in Path(const.INFORMATICUP_TEXT_DATASET).glob('**/*')]
    for file_path in file_path_list:
        with open(file_path, 'r') as file:
            text_data = file.read()
            data_list.append(preprocess_text(text_data))
    return data_list


def intrinsic_dim_dataset_to_csv(data_path: str | Path, subset_fraction: Optional[float]=1., include_prompt: bool=False):
    """
    It loads data subset for the intrinsic dimension dataset into a single csv file,
    with additional feature to load fraction of data subsets.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    dataset_file_paths = [entity for entity in data_path.glob('**/*') if entity.is_file()]

    dataset_file_paths_dict = OrderedDict()
    for file in dataset_file_paths:
        dataset_file_paths_dict[file] = file.name.split(".")[0].split("_")

    human_text_data_values, human_text_split_values, human_text_generator_values, human_text_source_values = [], [], [], []
    ai_gen_text_data_values, ai_gen_text_split_values, ai_gen_text_generator_values, ai_gen_text_source_values = [], [], [], []

    for (dataset_file_path, dataset_ext_list) in tqdm(dataset_file_paths_dict.items()):
        with open(dataset_file_path) as json_data_file:
            json_list = json.load(json_data_file)
            json_list = random.sample(json_list, int(subset_fraction*len(json_list)))
            for json_value in json_list:
                if include_prompt:
                    human_text_data_values.append(json_value['prefix']+" "+json_value['gold_completion'])
                else:
                    human_text_data_values.append(json_value['gold_completion'])
                human_text_split_values.append(json_value['split'])
                human_text_generator_values.append(dataset_ext_list[0])
                human_text_source_values.append(dataset_ext_list[-1])
                if include_prompt:
                    ai_gen_text_data_values.append(json_value['prefix']+" "+ ''.join(json_value['gen_completion']))
                else:
                    ai_gen_text_data_values.append(''.join(json_value['gen_completion']))
                ai_gen_text_split_values.append(json_value['split'])
                ai_gen_text_generator_values.append(dataset_ext_list[1])
                ai_gen_text_source_values.append(dataset_ext_list[-1])

    human_text_flag_values = [const.HUMAN_TEXT_STR]*len(human_text_data_values)
    ai_gen_text_flag_values = [const.AI_GENERATED_TEXT_STR]*len(ai_gen_text_data_values)

    human_text_data_df = pd.DataFrame(
        {
            
            'text_data' : human_text_data_values,
            'generator': human_text_generator_values,
            'data_source': human_text_source_values,
            'data_split' : human_text_split_values,
            'target_class': human_text_flag_values,
        })
    ai_gen_text_data_df = pd.DataFrame(
        {
            
            'text_data' : ai_gen_text_data_values,
            'generator': ai_gen_text_generator_values,
            'data_source': ai_gen_text_source_values,
            'data_split' : ai_gen_text_split_values,
            'target_class': ai_gen_text_flag_values,
        })
    
    trasformed_data_df = pd.concat([human_text_data_df, ai_gen_text_data_df], axis=0)
    trasformed_data_df.to_csv(Path(const.TRANSFORMED_DATA_SAVE_PATH) / const.TRANSFORMED_DATA_CSV_NAME, index=False)


def intrinsic_dim_gpt2_datasets_to_csv(data_path: str | Path, file_name: str):
    """
    It loads GPT2 generated datasets into different transformed csv files for training the intrinsic dimension model
    """
    data_df = pd.read_csv(Path(data_path) / file_name)
    data_sources = ["human"] * len(data_df['category'])
    data_sources.extend( ["gpt2"] * 3 * len(data_df['ai_text_dtr']))
    data_split_train = ["train"] * int(0.8 * 4 * len(data_df['category']))
    data_split_validation = ["validation"] * int(0.1 * 4 * len(data_df['category']))
    data_split_test = ["test"] * int( 4 * len(data_df['category']) - len(data_split_validation) - len(data_split_train) )
    data_split_final = data_split_train
    data_split_final.extend(data_split_validation)
    data_split_final.extend(data_split_test)
    data_split_final = random.sample(data_split_final, len(data_split_final))
    data_targets = [const.HUMAN_TEXT_STR] * len(data_df['category'])
    data_targets.extend( [const.AI_GENERATED_TEXT_STR] * 3 * len(data_df['ai_text_dtr']))
    data_texts = list(data_df['human_text'].apply(lambda x: preprocess_text(x)))
    data_texts.extend(list(data_df['ai_text_dtr'].apply(lambda x: preprocess_text(x))))
    data_texts.extend(list(data_df['ai_text_stc'].apply(lambda x: preprocess_text(x))))
    data_texts.extend(list(data_df['ai_text_con'].apply(lambda x: preprocess_text(x))))
    trasformed_data_df = pd.DataFrame(
        {
            'categories': 4 * list(data_df['category']),
            'text_data' : data_texts,
            'generator': data_sources,
            'data_source': data_sources,
            'data_split' : data_split_final,
            'target_class': data_targets,
        })
    trasformed_data_df.to_csv(Path(const.TRANSFORMED_DATA_SAVE_PATH) / str("intrinsic_dim_"+file_name), index=False)


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

def gpt2_dataset_to_files(data_path: str | Path, subset_fraction: Optional[float]=1.):
    """
    It loads GPT2 generated datasets into text files with data subset folders,
    with additional feature to load fraction of data subsets.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    csv_file_name = os.path.splitext(os.path.basename(data_path))[0]

    # Read the CSV file
    df = pd.read_csv(data_path)

    # Create the main directory and the 'ai' subdirectory
    main_dir = os.path.join(const.TRANSFORMED_DATA_SAVE_PATH, csv_file_name)
    ai_dir = os.path.join(main_dir, 'ai')
    os.makedirs(ai_dir, exist_ok=True)

    # Function to write text to a file
    def write_to_file(folder, filename, text):
        with open(os.path.join(folder, filename), 'w', encoding='utf-8') as file:
            file.write(preprocess_text(str(text)))

    # Iterate over the DataFrame and create text files
    for index, row in tqdm(df.iterrows()):
        write_to_file(ai_dir, f'ai_text_dtr_{index + 1}.txt', row['ai_text_dtr'])
        write_to_file(ai_dir, f'ai_text_stc_{index + 1}.txt', row['ai_text_stc'])
        write_to_file(ai_dir, f'ai_text_con_{index + 1}.txt', row['ai_text_con'])

    print('Text files created successfully.')


if __name__ == '__main__':
    # intrinsic_dim_dataset_to_csv(const.INTRINSIC_DIM_DATASET)
    # intrinsic_dim_dataset_to_files(const.INTRINSIC_DIM_DATASET)
    intrinsic_dim_gpt2_datasets_to_csv(const.TRANSFORMED_DATA_SAVE_PATH, f"{const.GEN_DE_DATA_CSV_NAME.split('.')[0]}_small.{const.GEN_DE_DATA_CSV_NAME.split('.')[1]}")
    intrinsic_dim_gpt2_datasets_to_csv(const.TRANSFORMED_DATA_SAVE_PATH, f"{const.GEN_DE_DATA_CSV_NAME.split('.')[0]}_large.{const.GEN_DE_DATA_CSV_NAME.split('.')[1]}")
    intrinsic_dim_gpt2_datasets_to_csv(const.TRANSFORMED_DATA_SAVE_PATH, const.GEN_EN_DATA_CSV_NAME)
