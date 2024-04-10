#!/bin/bash
#SBATCH --account=pavuskar
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

module purge
module load gcc/11.3.0 git/2.36.1 python/3.9.12
git clone https://github.com/nikitakodkany/csci566s-24.git
git checkout train
pip install -r requirements.txt 
python train.py --model transformer --run_name test-carc --batch_size 2 --epoch 32 --learning_rate 0.001 --device cpu --optimizer Adam --log_dir None --output_dir output --tweet_embedding mixedbread-ai/mxbai-embed-large-v1 --tweet_sentiment cardiffnlp/twitter-roberta-base-sentiment-latest --reddit_sentiment bhadresh-savani/distilbert-base-uncased-emotion --tweet_sector cardiffnlp/tweet-topic-latest-multi --seq_len 16 --stride 4 --num_heads 8 --num_layers 3 --d_model 1024 --dim_feedforward 2048 --num_outputs 3 --report_to_wandb --wandb_project dunes --wandb_entity pavuskarmihir --save_checkpoints_to_wandb
