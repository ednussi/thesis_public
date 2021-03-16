#!/usr/bin/env python
from data.squad_utils import *
from experiments.utils import *

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    squad_df = load_squad_df(squad_json_path)

    base_results_path = 'results/'
    exp_name = 'baseline'
    base_model = "roberta-base"
    csv_entery_num = 0

    model, tokenizer = get_model_tokenizer(base_model)
    csv_columns = ['EM', 'F1', 'seed', 'n', 'base_model']

    # Create results folder and opens log
    create_folder('results')
    create_folder(f'results/{exp_name}')
    create_folder(f'results/{exp_name}/data')

    with open(f'results/{exp_name}/results.csv', "a") as f:
        f.write(f',{",".join(csv_columns)}\n')

    for n in get_log_scale(max_power_range=2):
        for seed in range(1):

            # ============ Train ============
            # pick q-a-c triplets at random
            data_samples_df = sample_random_qa_pairs(squad_df, n, seed, exp_name)
            # prepare data for training
            train_ds = train_df_to_training_dataset(data_samples_df, tokenizer)
            model = train_model(model, tokenizer, train_ds)

            # ============ Eval ============
            res_summary, answers_df = eval_model(model, tokenizer)

            # ============ Save ============
            # save answers df
            answers_df_name = f'results/{exp_name}/data/answers-{n}_seed-{seed}.csv'
            answers_df.to_csv(answers_df_name)
            EM, F1 = [x.split(':')[1] for x in res_summary.split(',')]
            with open(f'results/{exp_name}/results.csv', "a") as myfile:
                save_string = f'{csv_entery_num},{EM},{F1},{seed},{n},{base_model}\n'
                myfile.write(save_string)
            csv_entery_num += 1