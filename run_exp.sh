#!/bin/sh
#SBATCH --time=2-0:0:0
#SBATCH --gres=gpu:rtx2080:1
python run_baseline.py --max_samples_exp 9 --exp_repeat 5 --name baseline
