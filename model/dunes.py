import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math

class DataLoaderDUNES(Dataset):
    def __init__(self, data, preprocessing_model, seq_len=5, stride=1):
        self.data = data
        self.preprocessing_model = preprocessing_model
        self.seq_len = seq_len
        self.stride = stride
        self.numeric_features = 6
        self.vector_size = sum(preprocessing_model.feature_size.values())+self.numeric_features
    
    def __len__(self):
        return max(0, ((len(self.data) - self.seq_len) // self.stride) + 1)
    
    def __getitem__(self, idx):
        seq_features = torch.zeros((self.seq_len, self.vector_size))
        start_idx = idx * self.stride

        for i in range(self.seq_len):
            data_idx = start_idx + i
            if data_idx < len(self.data):
                item = self.data[data_idx]
                if i == self.seq_len - 1:  # Masking for nth tweet
                    prev_tweet_embedding, prev_tweet_sentiment, prev_reddit_sentiment,positive_reddit_sentiment,negative_reddit_sentiment, prev_tweet_sector= self.preprocessing_model(item['tweet_content'], " "," "," ")
                    # Ensure prev_tweet_embedding is flattened to match dimensions
                    prev_tweet_embedding = prev_tweet_embedding.reshape(-1)
                    prev_tweet_embedding=torch.tensor(prev_tweet_embedding)
                    
                    prev_tweet_sentiment = prev_tweet_sentiment.reshape(-1)
                    prev_tweet_sentiment=torch.tensor(prev_tweet_sentiment) 
                    # Create a mask tensor with the appropriate length
                    mask_tensor = torch.zeros(self.vector_size - prev_tweet_embedding.size(0) - prev_tweet_sentiment.size(0))
                    all_features = torch.cat([prev_tweet_embedding, prev_tweet_sentiment, mask_tensor])
                else:
                    prev_tweet_embedding, prev_tweet_sentiment, prev_reddit_sentiment,positive_reddit_sentiment,negative_reddit_sentiment, prev_tweet_sector = self.preprocessing_model(item['tweet_content'], item['title_submission'],item['positive_comments'],item['negative_comments'])
                    # Flatten and concatenate all features
                    prev_tweet_embedding = prev_tweet_embedding.reshape(-1)
                    prev_tweet_embedding=torch.tensor(prev_tweet_embedding)
                    
                    prev_tweet_sentiment = prev_tweet_sentiment.reshape(-1)
                    prev_tweet_sentiment=torch.tensor(prev_tweet_sentiment)
                    
                    prev_reddit_sentiment = torch.tensor(prev_reddit_sentiment)
                    prev_reddit_sentiment = prev_reddit_sentiment.reshape(-1)
                    
                    positive_reddit_sentiment = torch.tensor(positive_reddit_sentiment)
                    positive_reddit_sentiment = positive_reddit_sentiment.reshape(-1)
                    
                    negative_reddit_sentiment = torch.tensor(negative_reddit_sentiment)
                    negative_reddit_sentiment = negative_reddit_sentiment.reshape(-1)
                    
                    prev_tweet_sector = prev_tweet_sector.reshape(-1)
                    prev_tweet_sector=torch.tensor(prev_tweet_sector)
                    additional_data = torch.tensor([item['score_submission'], item['positive_scores'], item['negative_scores'], item['num_likes'], item['num_retweets'], item['num_replies']], dtype=torch.float).view(-1)
                    all_features = torch.cat([prev_tweet_embedding, prev_tweet_sentiment, prev_reddit_sentiment, positive_reddit_sentiment, negative_reddit_sentiment, prev_tweet_sector, additional_data])
                
                seq_features[i, :all_features.size(0)] = all_features

        return seq_features.unsqueeze(0)

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