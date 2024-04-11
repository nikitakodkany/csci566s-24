import argparse
import os
import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import wandb

import torch
from torch import nn
from torch.optim import Adam

from utils import reformat_dataset, calculate_mean_absolute_error
from model.builder import create_preprocessing_model
from model.dunes import DataLoaderDUNES, DunesTransformerModel
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler

# create a scaler object
scaler = RobustScaler()

data = [
    {
        'prev_tweet': "@WholeMarsBlog Headline is misleading. Starlink can obviously offer far more robust positioning than GPS, as it will have ~1000X more satellites over time. Not all will have line of sight to users, but still >10X GPS & far stronger signal. Just not today’s problem.",
        'curr_tweet': "@spideycyp_155 @BillyM2k If Russia faced calamitous defeat in conventional warfare for something as strategically critical as Crimea, the probability of using nuclear weapons is high",
        'prev_reddit': "We know who controls the media. The same corporations who have wreaked havoc on the globe for decades, if not centuries, the big banks who financed them, and the governments who turned a blind eye to the destruction. The same entities who have brought us to the precipice of destruction - quite possibly condemning us, and our progeny to an unlivable climate They have tried to stop you at every turn, and yet you persist for the good of humanity. We love you, Elon! Keep up the good work! As you have said, we must never let the light of human consciousness fade - never!",
        'likes': 100,  
        'retweets': 50, 
        'comments': 25  
    },
    {
        'prev_tweet': "@WholeMarsBlog Headline is misleading. Starlink can obviously offer far more robust positioning than GPS, as it will have ~1000X more satellites over time. Not all will have line of sight to users, but still >10X GPS & far stronger signal. Just not today’s problem.",
        'curr_tweet': "@spideycyp_155 @BillyM2k If Russia faced calamitous defeat in conventional warfare for something as strategically critical as Crimea, the probability of using nuclear weapons is high",
        'prev_reddit': "We know who controls the media. The same corporations who have wreaked havoc on the globe for decades, if not centuries, the big banks who financed them, and the governments who turned a blind eye to the destruction. The same entities who have brought us to the precipice of destruction - quite possibly condemning us, and our progeny to an unlivable climate They have tried to stop you at every turn, and yet you persist for the good of humanity. We love you, Elon! Keep up the good work! As you have said, we must never let the light of human consciousness fade - never!",
        'likes': 100,  
        'retweets': 50, 
        'comments': 25  
    },
    {
        'prev_tweet': "@WholeMarsBlog Headline is misleading. Starlink can obviously offer far more robust positioning than GPS, as it will have ~1000X more satellites over time. Not all will have line of sight to users, but still >10X GPS & far stronger signal. Just not today’s problem.",
        'curr_tweet': "@spideycyp_155 @BillyM2k If Russia faced calamitous defeat in conventional warfare for something as strategically critical as Crimea, the probability of using nuclear weapons is high",
        'prev_reddit': "We know who controls the media. The same corporations who have wreaked havoc on the globe for decades, if not centuries, the big banks who financed them, and the governments who turned a blind eye to the destruction. The same entities who have brought us to the precipice of destruction - quite possibly condemning us, and our progeny to an unlivable climate They have tried to stop you at every turn, and yet you persist for the good of humanity. We love you, Elon! Keep up the good work! As you have said, we must never let the light of human consciousness fade - never!",
        'likes': 100,  
        'retweets': 50, 
        'comments': 25  
    },
    {
        'prev_tweet': "@WholeMarsBlog Headline is misleading. Starlink can obviously offer far more robust positioning than GPS, as it will have ~1000X more satellites over time. Not all will have line of sight to users, but still >10X GPS & far stronger signal. Just not today’s problem.",
        'curr_tweet': "@spideycyp_155 @BillyM2k If Russia faced calamitous defeat in conventional warfare for something as strategically critical as Crimea, the probability of using nuclear weapons is high",
        'prev_reddit': "We know who controls the media. The same corporations who have wreaked havoc on the globe for decades, if not centuries, the big banks who financed them, and the governments who turned a blind eye to the destruction. The same entities who have brought us to the precipice of destruction - quite possibly condemning us, and our progeny to an unlivable climate They have tried to stop you at every turn, and yet you persist for the good of humanity. We love you, Elon! Keep up the good work! As you have said, we must never let the light of human consciousness fade - never!",
        'likes': 100,  
        'retweets': 50, 
        'comments': 25  
    }
]

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='transformer', help='Model to train')
    parser.add_argument("--run_name", type=str, default="train-model", help="used to name saving directory and wandb run")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--device', type=str, default=None, help='GPU to use [default: none]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--lr_step_size', type=int, default=1, help='Step size for learning rate scheduler [default: 1]')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='Gamma for learning rate scheduler [default: 0.1]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--output_dir', type=str, default='output', help='Log path [default: None]')
    parser.add_argument('--data_path', type=str, default='dataset/elon_reddit_data.csv', help='Path to data file [default: dataset/elon_reddit_data.csv]')

    parser.add_argument('--tweet_embedding', type=str, default='mixedbread-ai/mxbai-embed-large-v1', help='Tweet embedding model')
    parser.add_argument('--tweet_sentiment', type=str, default='cardiffnlp/twitter-roberta-base-sentiment-latest', help='Tweet sentiment model')
    parser.add_argument('--reddit_sentiment', type=str, default='bhadresh-savani/distilbert-base-uncased-emotion', help='Reddit sentiment model')
    parser.add_argument('--tweet_sector', type=str, default='cardiffnlp/tweet-topic-latest-multi', help='Tweet sector model')

    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length for the model [default: 16]')
    parser.add_argument('--stride', type=int, default=4, help='Stride for the model [default: 4]')

    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for the transformer model [default: 8]')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers for the transformer model [default: 3]')
    parser.add_argument('--d_model', type=int, default=1024, help='Size of the model [default: 1024]')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Size of the feedforward network [default: 2048]')
    parser.add_argument('--num_outputs', type=int, default=3, help='Number of outputs [default: 3]')    

    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument( "--save_checkpoints_to_wandb", default=False, action="store_true", help="save checkpoints to wandb")
    
    return parser.parse_args()

def train_model(args, checkpoints_dir, output_dir):
    print("\nInitializing PreDunes Model")
    print("Tweet Embedding:", args.tweet_embedding)
    print("Tweet Sentiment:", args.tweet_sentiment)
    print("Reddit Sentiment:", args.reddit_sentiment)
    print("Tweet Sector:", args.tweet_sector)

    preprocessing_model = create_preprocessing_model(
        args.tweet_embedding,
        args.tweet_sentiment,
        args.reddit_sentiment,
        args.tweet_sector,
        args.device
    )

    print("\nLoading Data")
    print("Data Path:", args.data_path)
    df = pd.read_csv(args.data_path)
    df[['num_likes', 'num_retweets', 'num_replies']] = scaler.fit_transform(df[['num_likes', 'num_retweets', 'num_replies']])
    print("Target Stats\n")
    print(df[['num_likes', 'num_retweets', 'num_replies']].describe().loc[['mean', 'std', 'max', 'min']])
    data = reformat_dataset(df)
    dataset = DataLoaderDUNES(data, preprocessing_model, seq_len=args.seq_len, stride=args.stride)
    print("Data Loaded")
    print("Data Length:", len(dataset))
    print("Batch Size:", args.batch_size)
    print("Sequence Length:", args.seq_len)
    print("Stride:", args.stride)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\nInitializing DunesTransformerModel")
    print("Feature Sizes:", preprocessing_model.feature_size)
    print("model:", args.model)
    print("d_model:", args.d_model)
    print("nhead:", args.num_heads)
    print("num_encoder_layers:", args.num_layers)
    print("dim_feedforward:", args.dim_feedforward)
    print("num_outputs:", args.num_outputs)

    model = DunesTransformerModel(
        feature_size=sum(preprocessing_model.feature_size.values()),
        d_model=args.d_model,  # Size of each projection layer
        nhead=args.num_heads,  # Number of attention heads in the transformer encoder
        num_encoder_layers=args.num_layers,  # Number of layers in the transformer encoder
        dim_feedforward=args.dim_feedforward,  # Size of the feedforward network model in transformer encoder
        num_outputs=args.num_outputs  # Number of output values (e.g., predicting engagement metrics)
    ).to(args.device)

    print("\nTraining Model")
    print("Epochs:", args.epoch)
    print("Learning Rate:", args.learning_rate)
    print("Learning Rate Scheduler Step Size:", args.lr_step_size)
    print("Learning Rate Scheduler Gamma:", args.lr_gamma)
    print("Optimizer:", args.optimizer)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    num_epochs = args.epoch

    for epoch in range(num_epochs):
        # Training phase
        model.train()  
        train_loss = 0.0
        likes_mae = 0.0
        retweets_mae = 0.0
        replies_mae = 0.0
        for batch, targets in tqdm(dataloader, desc='batches', leave=False):
            optimizer.zero_grad()
            batch = batch.to(args.device)
            targets = targets.to(args.device)
            batch = batch.permute(1, 0, 2)
            outputs = model(batch)
            loss = criterion(outputs, targets)
            likes_mae += calculate_mean_absolute_error(outputs[:, 0], targets[:, 0])
            retweets_mae += calculate_mean_absolute_error(outputs[:, 1], targets[:, 1])
            replies_mae += calculate_mean_absolute_error(outputs[:, 2], targets[:, 2])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}\t Train Loss: {avg_train_loss:.4f}')

        if args.report_to_wandb: 
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"], 
                "loss": loss,
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "likes_mae": likes_mae,
                "retweets_mae": retweets_mae,
                "replies_mae": replies_mae
                },commit=True
            )

        if epoch%4 == 0:
            model.eval()
            val_loss = 0.0
            for val_batch, val_target in val_dataloader:
                val_batch = val_batch.permute(1, 0, 2)
                val_output = model(val_batch)
                loss = criterion(val_output, val_target)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f'\t\t Val Loss: {avg_val_loss:.4f}')
            
            save_path = Path.joinpath(checkpoints_dir, f'weights_{epoch}.pt')
            torch.save(model.state_dict(), save_path)
            
            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(save_path)
            if args.report_to_wandb:
                wandb.log({
                    "val_loss": avg_val_loss,
                    "epoch": epoch,
                    "likes_mae": likes_mae,
                    "retweets_mae": retweets_mae,
                    "replies_mae": replies_mae
                    },commit=True
                )
        scheduler.step()
        
    print("Training Complete")

    model.eval()
    save_path = Path.joinpath(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    if args.report_to_wandb and args.save_checkpoints_to_wandb:
        wandb.save(save_path)

def main():
    args = parse_args()

    if args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else torch.device(args.device)
    print(f"Device: {device}")
    
    ''' Set up logging directories'''  
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    output_dir = Path.joinpath(Path(args.output_dir), Path(args.run_name))
    output_dir.mkdir(exist_ok=True) 
    output_dir = Path.joinpath(output_dir, Path(timestr))
    output_dir.mkdir(exist_ok=True)

    checkpoints_dir = Path.joinpath(output_dir, Path('./weights/'))
    checkpoints_dir.mkdir(exist_ok=True)

    print(f"Saving weights to {checkpoints_dir}")
    print(f"Saving output to {output_dir}")

    train_model(args, checkpoints_dir, output_dir)


if __name__ == '__main__':
    main()

'''
test command
python train.py --model transformer --run_name train-model --batch_size 2 --epoch 32 --learning_rate 0.001 --device cpu --optimizer Adam --log_dir None --output_dir output --tweet_embedding mixedbread-ai/mxbai-embed-large-v1 --tweet_sentiment cardiffnlp/twitter-roberta-base-sentiment-latest --reddit_sentiment bhadresh-savani/distilbert-base-uncased-emotion --tweet_sector cardiffnlp/tweet-topic-latest-multi --seq_len 10 --stride 2 --num_heads 8 --num_layers 3 --d_model 1024 --dim_feedforward 2048 --num_outputs 3 --report_to_wandb --wandb_project dunes --wandb_entity pavuskarmihir --save_checkpoints_to_wandb
'''