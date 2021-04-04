#!/usr/bin/env python
from data.squad_utils import *
from experiments.utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Parsing Args..')
    parser.add_argument('--bpath', type=str, nargs='1', required=False,
                        help='base path', default='/cs/labs/gabis/ednussi/')
    parser.add_argument('--max_samples_exp', type=int, nargs='1', required=False , default=9,
                        help='The exponent used on a base 2 to determine max number of #samples takes')
    parser.add_argument('--exp_repeat', type=int, nargs='1', required=False, default=5,
                        help='Repeat experiment this number of times, use a different random seed for each exp')
    parser.add_argument('--name', type=str, nargs='1', required=True, help='Experiment name')


    args = parser.parse_args()
    return args

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def run_base():
    args = parse_args()
    squad_df = load_squad_df(squad_json_path)

    base_path = args.bpath
    exp_name = 'baseline'
    base_model = "roberta-base"
    csv_entery_num = 0

    model, tokenizer = get_model_tokenizer(base_model)
    csv_columns = ['EM', 'F1', 'seed', 'n', 'base_model']

    # Create results folder and opens log
    create_folder(f'{base_path}/results')
    create_folder(f'{base_path}/results/{exp_name}')
    create_folder(f'{base_path}/results/{exp_name}/data')

    results_path = f'{base_path}/results/{exp_name}/results.csv'
    with open(results_path, "a") as f:
        f.write(f',{",".join(csv_columns)}\n')

    for n in get_log_scale(max_power_range=args.max_samples_exp):
        for seed in range(args.exp_repeat):

            # ============ Train ============
            # pick q-a-c triplets at random
            data_samples_df = sample_random_qa_pairs(squad_df, n, seed, exp_name)
            # prepare data for training
            train_ds = train_df_to_training_dataset(data_samples_df, tokenizer, shuffle_seed=0)
            model = train_model(model, tokenizer, train_ds)

            # ============ Eval ============
            res_summary, answers_df = eval_model(model, tokenizer)

            # ============ Save ============
            # save answers df
            answers_df_name = f'{base_path}/results/{exp_name}/data/answers-{n}_seed-{seed}.csv'
            answers_df.to_csv(answers_df_name)
            EM, F1 = [x.split(':')[1] for x in res_summary.split(',')]
            with open(results_path, "a") as myfile:
                save_string = f'{csv_entery_num},{EM},{F1},{seed},{n},{base_model}\n'
                myfile.write(save_string)
            csv_entery_num += 1

    # ============ Plot Results ============
    plot_random_sample_res([results_path], exp_name, base_path)
    print('Experiment Run Successfully')

if __name__ == '__main__':
    run_base()