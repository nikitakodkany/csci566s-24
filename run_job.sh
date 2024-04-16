#!/bin/bash
#SBATCH --account=yzhao010_1246
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

module purge
module load gcc/11.3.0 git/2.36.1 python/3.9.12
cd csci566s-24
mkdir output
git checkout train
git pull origin train
pip install -r requirements.txt 
python train.py --model transformer --run_name log-transformer-1 --batch_size 32 --epoch 32 --learning_rate 0.0001 --optimizer Adam --lr_step_size 1 --lr_gamma 0.9 --loss mse --log_dir None --output_dir output --data_path dataset/elon_twitter_data.csv --logarithmic --validation_split 0.1 --tweet_embedding mixedbread-ai/mxbai-embed-large-v1 --tweet_sentiment cardiffnlp/twitter-roberta-base-sentiment-latest --reddit_sentiment bhadresh-savani/distilbert-base-uncased-emotion --tweet_sector cardiffnlp/tweet-topic-latest-multi --seq_len 16 --stride 4 --num_heads 8 --num_layers 3 --d_model 1024 --dim_feedforward 2048 --num_outputs 3 --report_to_wandb --wandb_project dunes --wandb_entity pavuskarmihir --save_checkpoints_to_wandb
