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
import matplotlib.pyplot as plt
from data.squad.eval1 import evaluate
import nlpaug.augmenter.word as naw
import os
import requests
import gzip
import shutil


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

# =================== Data Sampling ===================

def sample_random_qa_pairs(df, n, random_state, exp_name, bpath, augs, augs_count, augs_names_str):
    df_sample = df.sample(n=n, random_state=random_state)

    # Add augmentations if needed
    df_sample = add_augs_to_df(df_sample, augs=augs, augs_count=augs_count)

    # Save data
    sample_df_name = f'{bpath}/results/{exp_name}/data/samples-{n}_seed-{random_state}{augs_names_str}.csv'
    df_sample.to_csv(sample_df_name, index=False)
    print(f'Saved data random sample {sample_df_name}')

    return df_sample

# =================== Adding Augs Functions ===================

def get_n_new_aug(aug, text, new_augs_num):
    news_augs_set = {aug.augment(text) for i in range(new_augs_num)}
    while len(news_augs_set) < new_augs_num:
        news_augs_set.update({aug.augment(text)})
    return news_augs_set

def augs_from_df_row(df_row, aug, new_augs_num):
    q = df_row['question']
    a = df_row['answer']
    qa_id = df_row['id']
    title = df_row['title']
    context = df_row['context']

    news_q_augs_set = get_n_new_aug(aug, q, new_augs_num)
    qa_pair = {'title':title,'context':context,'answer': a, 'question': q, 'id': qa_id}
    new_aug_qa_list = [qa_pair]
    for i, new_aug_q in enumerate(news_q_augs_set):
        new_aug_qa_list.append({'title':title,'context':context,'answer': a, 'question': new_aug_q, 'id': qa_id + f'_{i + 1}'})
    return new_aug_qa_list

def aug_from_df(df, aug, new_augs_num):
    news_augs_qa_list = []
    for index, row in df.iterrows():
        news_augs_qa_list.extend(augs_from_df_row(row, aug, new_augs_num))
    return news_augs_qa_list

def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def get_augs_from_names(augs_names):
    googlenews_bin = 'data/GoogleNews-vectors-negative300.bin'
    if not os.path.exists(googlenews_bin):
        print('Downloading GoogleNews-vectors-negative300.bin.gz')
        googlenews_bin_gz_url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

        download(googlenews_bin_gz_url, f'{googlenews_bin}.gz')
        print('Unzipping gz file')
        # un-gz file and save as bin
        with gzip.open(f'{googlenews_bin}.gz', 'rb') as f_in:
            with open(googlenews_bin, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print('Done')

    print('Loading Augs')
    switcher = {
        'delete-random': naw.RandomWordAug(),
        'insert-word-embed':naw.WordEmbsAug(model_type='word2vec',
                                model_path=googlenews_bin,
                                action="insert"),
        'sub-word-embed': naw.WordEmbsAug(model_type='word2vec',
                                model_path=googlenews_bin,
                                action="substitute"),
        'insert-bert-embed': naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert"),
        'sub-bert-embed': naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    }

    return [switcher.get(aug_name, None) for aug_name in augs_names]

def add_augs_to_df(df, augs, augs_count):
    # augs is empty or just augs_count is 0
    if not augs:
        return df

    news_augs_qa_list = []
    for aug in augs:
        news_augs_qa_list.extend(aug_from_df(df, aug, new_augs_num=augs_count))
    augs_df = pd.DataFrame.from_dict(news_augs_qa_list)
    augs_df = augs_df.drop_duplicates(subset=['id'])
    return augs_df

# =================== Train Functions ===================

def train_model(model, train_ds, train_log_path):

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
    csv_entery_num =0
    csv_columns = ['epoch', 'batch', 'loss']
    with open(train_log_path, "a") as f:
        f.write(f',{",".join(csv_columns)}\n')

        # Train loop
        for epoch_i in tqdm(range(epochs), desc='Train Epochs'):
            for batch_i, batch in enumerate(batches):
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
                save_string = f'{csv_entery_num},{epoch_i},{batch_i},{loss}\n'
                f.write(save_string)
                csv_entery_num += 1

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

# def get_em_f1_squad1(pred_path, bpath):
#     # pred_path = 'data/squad/sub_ex.json' # default test file
#     ground_truth_path = f'{bpath}/data/squad/dev-v1.1.json'
#     eval_script_path = f'{bpath}/data/squad/eval1.py' #v1
#     cmd = f'python {eval_script_path} {ground_truth_path} {pred_path}'
#     proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=True)
#     (out, err) = proc.communicate()
#     out = out.decode("utf-8")
#     EM, F1 = [float(out[x.start():x.end()]) for x in re.finditer('\d+\.\d+', out)]
#     return EM, F1

def run_eval1(pred_path, bpath):
    # Matching the Main function in eval1 .py
    #  python data/squad/eval1.py data/squad/squad1.1/dev-v1.1.json results.txt

    expected_version = '1.1'
    ground_truth_path = f'{bpath}/data/squad/dev-v1.1.json'
    eval_script_path = f'{bpath}/data/squad/eval1.py' #v1

    with open(ground_truth_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(pred_path) as prediction_file:
        predictions = json.load(prediction_file)

    em_f1_dict = evaluate(dataset, predictions)
    em = em_f1_dict['exact_match']
    f1 = em_f1_dict['f1']
    return em, f1

def calc_scores(answers_df, bpath):
    # Get a temp json file  of answers from df
    answers_json = pd.Series(answers_df.pred.values, index=answers_df.id).to_dict()
    tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    json.dump(answers_json, tfile)
    tfile.flush()
    EM, F1 = run_eval1(tfile.name, bpath)
    res_summary = f"EM:{EM}, F1:{F1}"
    return res_summary

def eval_model(model, tokenizer, bpath):
    answers_df = create_answers(model, tokenizer)
    # Get F1/EM scores using official code
    res_summary = calc_scores(answers_df, bpath)
    return res_summary, answers_df

def plot_random_sample_res(res_csv_paths, exp_name, bpath):

    plt.figure(figsize=(20,30))
    fig, (ax_f1, ax_em) = plt.subplots(2)
    ax_em.set_xscale("log")
    ax_f1.set_xscale("log")
    fig.suptitle('Experiment Results')

    for df_i,df_path in enumerate(res_csv_paths):
        df = pd.read_csv(df_path)
        rseed = df['seed'].unique()[0]
        x = []
        y_f1 = []
        y_em = []
        yerr_f1_min = []
        yerr_em_min = []
        yerr_f1_max = []
        yerr_em_max = []

        for context_limit in df['n'].unique():
            df_aug_cont_lim = df[df['n'] == context_limit]
            x.append(context_limit)
            y_f1.append(df_aug_cont_lim['F1'].mean())
            y_em.append(df_aug_cont_lim['EM'].mean())

            yerr_f1_min.append(df_aug_cont_lim['F1'].min())
            yerr_em_min.append(df_aug_cont_lim['EM'].min())
            yerr_f1_max.append(df_aug_cont_lim['F1'].max())
            yerr_em_max.append(df_aug_cont_lim['EM'].max())

        ax_f1.plot(x, y_f1, label=f'exp_{rseed}')
        ax_f1.fill_between(x, yerr_f1_min, yerr_f1_max, alpha=0.5)
        ax_em.plot(x, y_em, label=f'exp_{rseed}')
        ax_em.fill_between(x, yerr_em_min, yerr_em_max, alpha=0.5)


    xrange = [2 ** x for x in range(1, 9)]
    xrange_text = [str(x) for x in xrange]

    ax_f1.set_title('F1 vs. #QA pairs')
    ax_f1.legend(prop={'size':6}, loc='upper left')
    ax_f1.set_xticks(xrange)
    ax_f1.set_xticklabels(xrange_text)
    ax_em.set_title('EM vs. #QA pairs')
    ax_em.legend(prop={'size':6}, loc='upper left')
    ax_em.set_xticks(xrange)
    ax_em.set_xticklabels(xrange_text)
    fig.tight_layout()
    fig_save_path = f'{bpath}/results/{exp_name}/plot.png'
    fig.savefig(fig_save_path)
    print(f'Saved plot figure in {fig_save_path}')

if __name__ == '__main__':
    pass