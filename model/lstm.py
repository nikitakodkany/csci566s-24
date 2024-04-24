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
        self.vector_size = sum(preprocessing_model.feature_size.values())
    
    def __len__(self):
        return max(0, ((len(self.data) - self.seq_len) // self.stride) + 1)
    
    def __getitem__(self, idx):
        seq_features = torch.zeros((self.seq_len, self.vector_size))
        target_features = None
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
                    target_features = torch.tensor([item['num_likes'], item['num_replies'], item['num_retweets']], dtype=torch.float).view(-1)
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
                    additional_data[torch.isnan(additional_data)] = 0
                    all_features = torch.cat([prev_tweet_embedding, prev_tweet_sentiment, prev_reddit_sentiment, positive_reddit_sentiment, negative_reddit_sentiment, prev_tweet_sector, additional_data])
                
                seq_features[i, :all_features.size(0)] = all_features

        return seq_features, target_features

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer to produce the final outputs
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the output of the last time step to the dropout layer
        out = self.dropout(out[:, -1, :])
        
        # Get final outputs
        out = self.fc(out)
        
        return out