#!/usr/bin/env python
import urllib.request
import os
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path
import ast
from transformers import AutoTokenizer
import torch

squad_json_path = "data/squad/train-v1.1.json"

def download_squad():
    # check if squad already exists

    squad_path = Path("data/squad")
    if not squad_path.exists():
        print('============= Downloading Squad Data =============')
        os.makedirs(squad_path)

        # official site https://rajpurkar.github.io/SQuAD-explorer/
        # download from official git repo https://github.com/rajpurkar/SQuAD-explorer
        dev1_url = "https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json"
        dev2_url = "https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v2.0.json"
        train1_url = "https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/train-v1.1.json"
        train2_url = "https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/train-v2.0.json"
        urls = [dev1_url, dev2_url, train1_url, train2_url]

        for data_url in tqdm(urls, desc='Downloading Squad'):
            fname = data_url.split('/')[-1]
            with urllib.request.urlopen(dev1_url) as url:
                data = json.loads(url.read().decode())

            with open(squad_path/fname, 'w') as outfile:
                json.dump(data, outfile)
        print('============= Done Downloading Squad Data =============')

def squad1_to_df(squad_json_path):

    #if doesn't exist download it
    download_squad()
    squad_json_path = Path(squad_json_path)
    with open(squad_json_path, 'r') as f:
        squad_json = json.load(f)

    dict_list =[]

    for group_i, group in tqdm(enumerate(squad_json['data']), desc='Titles'):  # 442 paras
        title = group['title']
        for passage_i, passage in enumerate(group['paragraphs']):  # Total of 18896 contexts
            context = passage['context']
            for qa_pair in passage['qas']:
                question = qa_pair['question']
                id = qa_pair['id']
                for answer in qa_pair['answers']:
                    dict_list.append({'title':title, 'context':context, 'id':id, 'question':question, 'answer':answer })

    squad_df = pd.DataFrame.from_dict(dict_list)
    squad_df_csv_path = squad_json_path.with_suffix('.csv')
    # the csv is the infalted json file to save time when traversing on q-a-c pairs
    squad_df.to_csv(squad_df_csv_path, index=False)
    return squad_df

def load_squad_df(squad_json_path):
    squad_json_path = Path(squad_json_path)
    squad_df_csv_path = squad_json_path.with_suffix('.csv')
    print(squad_df_csv_path)
    if squad_df_csv_path.exists():
        print('loading csv')
        squad_df = pd.read_csv(squad_df_csv_path)
        # turn back answers column from string to dict
        squad_df['answer'] = squad_df['answer'].apply(ast.literal_eval)
    else:
        print('creating csv')
        squad_df = squad1_to_df(squad_json_path)
    return squad_df

def sample_random_qa_pairs(df, n, random_state, exp_name, bpath):
    df_sample = df.sample(n=n, random_state=random_state)
    sample_df_name = f'{bpath}/results/{exp_name}/data/data-{n}_seed-{random_state}.csv'
    df_sample.to_csv(sample_df_name, index=False)
    print(f'Saved data random sample {sample_df_name}')
    return df_sample


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
        if end_positions[-1] is None:
            # end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# based on https://huggingface.co/transformers/custom_datasets.html#question-answering-with-squad-2-0
def train_df_to_training_dataset(train_df, tokenizer, shuffle_seed=None):
    if shuffle_seed:
        train_df = train_df.sample(frac=1, random_state=shuffle_seed)

    # turn to lists
    train_answers = train_df['answer'].to_list()
    train_questions = train_df['question'].to_list()
    train_contexts = train_df['context'].to_list()

    print(f'Done\nAdding end indicies..')
    add_end_idx(train_answers, train_contexts)
    print('Done\nApplying Tokenizer..')
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    print('Done\nAdding Tokens Positions..')
    add_token_positions(train_encodings, train_answers, tokenizer)
    print('Done\nTurning to Torch dataset..')
    train_dataset = SquadDataset(train_encodings)
    return train_dataset


if __name__ == '__main__':
    pass