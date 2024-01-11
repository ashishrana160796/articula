import nltk
import random
import pandas as pd
import models.gpt_generators as gen_model
from nltk.corpus import brown
from pathlib import Path
from common import constants as const
from common.data_utils import preprocess_text


random.seed(16)
nltk.download('brown')


def create_base_en_corpus(save_path: str | Path, file_name: str):
    text_categories = brown.categories()
    category_list, context_text_list, completion_text_list = [], [], []
    for category in text_categories:
        for file_id in brown.fileids(categories=category):
            category_list.append(category)
            context_len = random.choice(list(range(2, 6))) # appending end of article sentences
            context_text_list_data = brown.sents(file_id)[-2-context_len:-2]
            completion_text_list_data = brown.sents(file_id)[-2:]
            context_text = [' '.join(sent) for sent in context_text_list_data]
            context_text = preprocess_text(' '.join(context_text))
            completion_text = [' '.join(sent) for sent in completion_text_list_data]
            completion_text = preprocess_text(' '.join(completion_text))
            context_text_list.append(context_text)
            completion_text_list.append(completion_text)

            category_list.append(category)
            start_context_len = random.choice(list(range(2, 6))) # appending start of article sentences
            start_context_text_list_data = brown.sents(file_id)[2:2+start_context_len]
            start_completion_text_list_data = brown.sents(file_id)[2+start_context_len:4+start_context_len]
            start_context_text = [' '.join(sent) for sent in start_context_text_list_data]
            start_context_text = preprocess_text(' '.join(start_context_text))
            start_completion_text = [' '.join(sent) for sent in start_completion_text_list_data]
            start_completion_text = preprocess_text(' '.join(start_completion_text))
            context_text_list.append(start_context_text)
            completion_text_list.append(start_completion_text)

    brown_data_df = pd.DataFrame(
        {
            
            'category' : category_list,
            'prompt_text': context_text_list,
            'human_text': completion_text_list,
        })

    brown_data_df.to_csv(Path(save_path) / file_name, index=False)


def generate_gpt_en_text(data_path: str | Path, file_name: str):
    data_df = pd.read_csv(Path(data_path) / file_name)
    data_df["ai_text_dtr"] = gen_model.GPT2Generator(list(data_df["prompt_text"]), min_length=8, max_length=140).generate_text()
    data_df["ai_text_stc"] = gen_model.GPT2Generator(list(data_df["prompt_text"]), min_length=8, max_length=140, gen_method='stc').generate_text()
    data_df["ai_text_con"] = gen_model.GPT2Generator(list(data_df["prompt_text"]), min_length=8, max_length=140, gen_method='con').generate_text()
    data_df.to_csv(Path(data_path) / file_name, index=False)

if __name__ == '__main__':
    create_base_en_corpus(const.TRANSFORMED_DATA_SAVE_PATH, const.GEN_EN_DATA_CSV_NAME)
    generate_gpt_en_text(const.TRANSFORMED_DATA_SAVE_PATH, const.GEN_EN_DATA_CSV_NAME)
