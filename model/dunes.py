import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math

class DataLoaderDUNES(Dataset):
    def __init__(self, data, preprocessing_model):
        self.data = data
        self.preprocessing_model = preprocessing_model
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prev_tweet_embedding, prev_tweet_sentiment, prev_reddit_sentiment, prev_tweet_sector = self.preprocessing_model(
            item['prev_tweet'], item['prev_reddit']
        )

        prev_tweet_embedding, prev_tweet_sentiment, prev_reddit_sentiment, prev_tweet_sector = self.preprocessing_model(
            item['prev_tweet'], item['prev_reddit']
        )

        def to_tensor(obj):
            if not isinstance(obj, torch.Tensor):
                obj = torch.tensor(obj, dtype=torch.float)
            return obj

        prev_tweet_embedding = to_tensor(prev_tweet_embedding)
        prev_tweet_sentiment = to_tensor(prev_tweet_sentiment)
        prev_reddit_sentiment = to_tensor(prev_reddit_sentiment)
        prev_tweet_sector = to_tensor(prev_tweet_sector)

        return {
            'prev_tweet_embedding': prev_tweet_embedding.squeeze(),
            'prev_tweet_sentiment': prev_tweet_sentiment,
            'prev_reddit_sentiment': prev_reddit_sentiment,
            'prev_tweet_sector': prev_tweet_sector,
            'likes': torch.tensor(item['likes'], dtype=torch.float),
            'retweets': torch.tensor(item['retweets'], dtype=torch.float),
            'comments': torch.tensor(item['comments'], dtype=torch.float)
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DunesTransformerModel(nn.Module):
    def __init__(self, feature_sizes, d_model, nhead, num_encoder_layers, dim_feedforward, num_outputs):
        super(DunesTransformerModel, self).__init__()
        self.positional_encoder = PositionalEncoding(d_model)
        self.fc1 = nn.Linear(sum(feature_sizes.values()), d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(d_model, num_outputs)

    def forward(self, features):
        # Concatenation of all features to create the input tensor
        src = torch.cat([features[key] for key in features], dim=1)
        src = self.fc1(src)
        src = self.fc2(src)
        src = self.positional_encoder(src)
        
        # Transformer encoding and output processing remain unchanged
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.output_linear(output)
        
        return output