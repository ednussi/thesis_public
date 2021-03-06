                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               #!/usr/bin/env python
from data.squad_utils import *
from experiments.utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Parsing Args..')
    parser.add_argument('--bpath', type=str, nargs='?', required=False,
                        help='base path', default='/cs/labs/gabis/ednussi/thesis_public')
    parser.add_argument('--max_samples_exp', type=int, nargs='?', required=False , default=9,
                        help='The exponent used on a base 2 to determine max number of #samples takes')
    parser.add_argument('--exp_repeat', type=int, nargs='?', required=False, default=5,
                        help='Repeat experiment this number of times, use a different random seed for each exp')
    parser.add_argument('--base_model', type=str, nargs='?', required=False, default="roberta-base",
                        help='Base tokenizer + architecture to use from huggingface models')
    parser.add_argument('--max_shuffle_repeat', type=int, nargs='?', required=False, default=1,
                        help='Given a set of examples, number of times to run on different shuffles order of examples ')
    parser.add_argument('--augs_names', nargs='+', required=False, default=[],
                        help='list of augs name from: delete-random, insert-word-embed, sub-word-embed, insert-bert-embed, sub-bert-embed')
    parser.add_argument('--augs_count', type=int, nargs='?', required=False, default=0,
                        help='number of aug to add per sample')
    parser.add_argument('--plot_res', required=False, action='store_true', default=False,
                        help='plot and save results')
    parser.add_argument('--lr', type=float, nargs='?', required=False, default=3e-5,
                        help='learning rate')
    parser.add_argument('--adam_eps', type=float, nargs='?', required=False, default=1e-8,
                        help='Epsilon in AdamW optimizer')
    parser.add_argument('--warmup_ratio', type=float, nargs='?', required=False, default=0.1,
                        help='warmup ratio during training')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')

    args = parser.parse_args()
    return args

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def run_base():
    args = parse_args()
    squad_df = load_squad_df(squad_json_path)

    base_path = args.bpath
    exp_name = args.name
    base_model = args.base_model
    csv_entery_num = 0

    model, tokenizer = get_model_tokenizer(base_model)
    csv_columns = ['EM', 'F1', 'seed', 'shuffle_seed', 'n', 'base_model']

    # Create results folder and opens log
    create_folder(f'{base_path}/results')
    create_folder(f'{base_path}/results/{exp_name}')
    create_folder(f'{base_path}/results/{exp_name}/data')

    results_path = f'{base_path}/results/{exp_name}/results.csv'
    with open(results_path, "a") as f:
        f.write(f',{",".join(csv_columns)}\n')

    # turn augs name list to
    augs =[]
    augs_names_str = ''
    if args.augs_count and len(args.augs_names):
        augs = get_augs_from_names(args.augs_names)
        augs_names_str = '_augs-' + '-'.join(args.augs_names)

    for n in get_log_scale(max_power_range=args.max_samples_exp):
        for seed in range(args.exp_repeat):

            # ============ Train ============
            # pick q-a-c triplets at random
            data_samples_df = sample_random_qa_pairs(squad_df, n, seed, exp_name, base_path,
                                                     augs, args.augs_count, augs_names_str)

            # prepare data for training
            for shuffle_seed in range(args.max_shuffle_repeat):
                exp_params = f'samples-{n}_seed-{seed}_shuffleseed-{shuffle_seed}{augs_names_str}'

                train_ds = train_df_to_training_dataset(data_samples_df, tokenizer, shuffle_seed=shuffle_seed)
                train_log_path = f'{base_path}/results/{exp_name}/trainlog_{exp_params}.csv'
                model = train_model(model, train_ds, train_log_path,
                                    args.lr, args.adam_eps, args.warmup_ratio)

                # ============ Eval ============
                res_summary, answers_df = eval_model(model, tokenizer, base_path)

                # ============ Save ============
                # save answers df
                answers_df_name = f'{base_path}/results/{exp_name}/data/answers_{exp_params}.csv'
                answers_df.to_csv(answers_df_name)
                EM, F1 = [x.split(':')[1] for x in res_summary.split(',')]
                with open(results_path, "a") as myfile:
                    save_string = f'{csv_entery_num},{EM},{F1},{seed},{shuffle_seed},{n},{base_model}\n'
                    myfile.write(save_string)
                csv_entery_num += 1

    # ============ Plot Results ============
    if args.plot_res:
        plot_random_sample_res([results_path], exp_name, base_path)
        print('Experiment Run Successfully')


if __name__ == '__main__':
    # python run_baseline.py --max_samples_exp 9 --exp_repeat 5 --name baseline
    # python run_baseline.py --max_samples_exp 9 --exp_repeat 1 --name shuffle_fixed_samples --max_shuffle_repeat 5
    # python run_baseline.py --max_samples_exp 9 --exp_repeat 5 --name aug1 --max_shuffle_repeat 1 --augs_names delete-random --augs_count 4

    run_base()