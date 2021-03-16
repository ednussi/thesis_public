#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader
import math
import pandas as pd
import json
import sys
import string
import tempfile
import subprocess
import re

def get_log_scale(max_power_range=9):
    # max_power_range=9 results in
    # [2, 4, 8, 16, 32, 64]
    log_scale_nums = [2 ** x for x in range(1, max_power_range)]
    return log_scale_nums

def get_model_tokenizer(base_model):
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(base_model)
    return model, tokenizer

# =================== Train Functions ===================

def train_model(model, tokenizer, train_ds):

    print(f'Cuda Visible: {torch.cuda.is_available()}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # Add model to device
    device = "cuda:0"
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    # optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    # Remove randomness from batches for reproducibility
    torch.manual_seed(0) #TODO
    batches = [x for x in tqdm(train_loader)]

    # matching scheme Few-Shot Question Answering by Pretraining Span Selection
    # https://arxiv.org/pdf/2101.00438.pdf
    # Need to chose max(200 steps, 10 epochs)
    # A step is one update meaning one batch
    epochs_to_reach_200 = math.ceil(200 * 1.0 / len(batches))
    epochs = max(10, epochs_to_reach_200)

    # Train loop
    for _ in tqdm(range(epochs), desc='Train Epochs'):
        for batch in batches:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Avoiding cuda errors
            input_ids = input_ids[:,:512]
            attention_mask = attention_mask[:,:512]

            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

    return model

# =================== Evals Funtions ===================

def get_squad_dev1_dataset():
    base_path = 'data/squad/'
    expected_version = '1.1'
    dataset_file = base_path + 'dev-v1.1.json'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    return dataset

def get_answer(model, tokenizer, context, question):
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    input_ids = input_ids[:512]
    attention_mask = attention_mask[:512]

    # Add inputs to device
    input = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])

    # move to gpu
    device = "cuda:0"
    input = input.to(device)
    attention_mask = attention_mask.to(device)

    answer_output = model(input, attention_mask=attention_mask)
    start_scores = answer_output.start_logits
    end_scores = answer_output.end_logits
    ans_tokens = input_ids[torch.argmax(start_scores): torch.argmax(end_scores) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    return  answer_tokens_to_string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def q_a_score(pred, gts):
    scores_for_ground_truths = []
    for ground_truth in gts:
        score = normalize_answer(pred) == normalize_answer(ground_truth['text'])
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def create_answers(model, tokenizer):
    scores_df = pd.DataFrame(columns=['id', 'q', 'pred', 'gt', 'score'])
    dev_dataset = get_squad_dev1_dataset()
    for article in tqdm(dev_dataset, desc='Eval Articles'):
        for paragraph in tqdm(article['paragraphs'], desc='Eval Paras'):
            context = paragraph['context']
            for qas in paragraph['qas']:
                qa_id = qas['id']
                q = qas['question']
                a = qas['answers']
                pred = get_answer(model=model, tokenizer=tokenizer, context=context, question=q)
                score = q_a_score(pred, a)
                # check score for mismatch
                scores_df = scores_df.append({'id': qa_id, 'q': q, 'pred': pred, 'gt': a, 'score': score},
                                             ignore_index=True)
    return scores_df

def get_em_f1_squad1(pred_path):
    # pred_path = 'data/squad/sub_ex.json' # default test file
    ground_truth_path = 'data/squad/dev-v1.1.json'
    eval_script_path = 'data/squad/eval1.py' #v1
    cmd = f'python {eval_script_path} {ground_truth_path} {pred_path}'
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode("utf-8")
    EM, F1 = [float(out[x.start():x.end()]) for x in re.finditer('\d+\.\d+', out)]
    return EM, F1

def calc_scores(answers_df):

    # Get a temp json file  of answers from df
    answers_json = pd.Series(answers_df.pred.values, index=answers_df.id).to_dict()
    tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    json.dump(answers_json, tfile)
    tfile.flush()
    EM, F1 = get_em_f1_squad1(tfile.name)
    res_summary = f"EM:{EM}, F1:{F1}"
    return res_summary

def eval_model(model, tokenizer):
    answers_df = create_answers(model, tokenizer)
    # Get F1/EM scores using official code
    res_summary = calc_scores(answers_df)
    return res_summary, answers_df

if __name__ == '__main__':
    pass