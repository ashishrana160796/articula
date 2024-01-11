import random
import pandas as pd
import models.gpt_generators as gen_model
from pathlib import Path
from common import constants as const
from common.data_utils import preprocess_text
import models.gpt_generators as gen_model

random.seed(16)


def create_base_de_corpus(data_path: str | Path, input_file_name: str, save_path: str | Path, output_file_name: str):
    de_data_df = pd.read_csv(
                        Path(data_path) / input_file_name,
                        sep=';',
                        on_bad_lines='skip',
                        names=['category', 'news_text'],
                        header=None
                    )

    small_de_data_df = de_data_df.drop(de_data_df[(de_data_df.news_text.map(lambda x: str(x).split(" ")).map(len) > 256)].index)
    large_de_data_df = de_data_df.drop(de_data_df[(de_data_df.news_text.map(lambda x: str(x).split(" ")).map(len) < 256) |
                                    (de_data_df.news_text.map(lambda x: str(x).split(" ")).map(len) > 768)].index)
    

    small_de_data_df['prompt_text'] = small_de_data_df.news_text.apply(lambda x: preprocess_text(" ".join(str(x).split(" ")[:len(str(x).split(" "))//2])))
    small_de_data_df['human_text'] = small_de_data_df.news_text.apply(lambda x: preprocess_text(" ".join(str(x).split(" ")[len(str(x).split(" "))//2:])))
    large_de_data_df['prompt_text'] = large_de_data_df.news_text.apply(lambda x: preprocess_text(" ".join(str(x).split(" ")[:len(str(x).split(" "))//2])))
    large_de_data_df['human_text'] = large_de_data_df.news_text.apply(lambda x: preprocess_text(" ".join(str(x).split(" ")[len(str(x).split(" "))//2:])))


    small_de_data_df = small_de_data_df.sample(frac=.3)
    large_de_data_df = large_de_data_df.sample(frac=.3)
    small_de_data_df.to_csv(Path(save_path) / f"{output_file_name.split('.')[0]}_small.{output_file_name.split('.')[1]}", index=False)
    large_de_data_df.to_csv(Path(save_path) / f"{output_file_name.split('.')[0]}_large.{output_file_name.split('.')[1]}", index=False)


def generate_gpt_de_text(data_path: str | Path, file_name: str, min_length: int, max_length: int):
    data_df = pd.read_csv(Path(data_path) / file_name)
    data_df["ai_text_dtr"] = [ preprocess_text(gen_text) for gen_text in gen_model.GPT2Generator(list(data_df["prompt_text"]), language='de', min_length=min_length, max_length=max_length).generate_text()]
    data_df["ai_text_stc"] = [ preprocess_text(gen_text) for gen_text in gen_model.GPT2Generator(list(data_df["prompt_text"]), language='de', min_length=min_length, max_length=max_length, gen_method='stc').generate_text()]
    data_df["ai_text_con"] = [ preprocess_text(gen_text) for gen_text in gen_model.GPT2Generator(list(data_df["prompt_text"]), language='de', min_length=min_length, max_length=max_length, gen_method='con').generate_text()]
    data_df.to_csv(Path(data_path) / file_name, index=False)


if __name__ == '__main__':
    create_base_de_corpus(const.DATA_PATH, const.DE_DATA_CSV_NAME, const.TRANSFORMED_DATA_SAVE_PATH, const.GEN_DE_DATA_CSV_NAME)
    generate_gpt_de_text(const.TRANSFORMED_DATA_SAVE_PATH, f"{const.GEN_DE_DATA_CSV_NAME.split('.')[0]}_small.{const.GEN_DE_DATA_CSV_NAME.split('.')[1]}", 64, 196)
    generate_gpt_de_text(const.TRANSFORMED_DATA_SAVE_PATH, f"{const.GEN_DE_DATA_CSV_NAME.split('.')[0]}_large.{const.GEN_DE_DATA_CSV_NAME.split('.')[1]}", 196, 512)
