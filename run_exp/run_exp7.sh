#!/bin/sh
#SBATCH --time=2-0:0:0
#SBATCH --gres=gpu:rtx2080:1
source  /cs/labs/gabis/ednussi/v1/bin/activate
python run_baseline.py --max_samples_exp 9 --exp_repeat 5 --name sub-bert-embed --augs_names sub-bert-embed --augs_count 4


