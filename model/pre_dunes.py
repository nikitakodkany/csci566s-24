import torch
import torch.nn as nn
import pandas as pd

class PreDUNES(nn.Module):
    def __init__(
            self, 
            twitter_embedding_model: nn.Module, 
            twitter_sentiment_tokenizer,
            twitter_sentiment_model: nn.Module,
            reddit_sentiment_tokenizer, 
            reddit_sentiment_model: nn.Module, 
            twitter_sector_tokenizer, 
            twitter_sector_model: nn.Module
        ):
        '''
        Initialize a DUNES model from a set of embeddings and sentiment models.
        Args:
            twitter_embedding_model: huggingface model for Twitter embeddings
            twitter_sentiment_model: huggingface model for Twitter sentiment
            reddit_sentiment_model: huggingface model for Reddit sentiment
            twitter_sector_model: huggingface model for Twitter sector
        '''
        super(PreDUNES, self).__init__()
        self.twitter_embedding_model = twitter_embedding_model
        self.twitter_sentiment_tokenizer = twitter_sentiment_tokenizer
        self.twitter_sentiment_model = twitter_sentiment_model
        self.reddit_sentiment_tokenizer = reddit_sentiment_tokenizer
        self.reddit_sentiment_model = reddit_sentiment_model
        self.twitter_sector_tokenizer = twitter_sector_tokenizer
        self.twitter_sector_model = twitter_sector_model
        self.feature_size = {
            'twitter_embedding': twitter_embedding_model.get_sentence_embedding_dimension(),
            'twitter_sentiment': twitter_sentiment_model.config.num_labels,
            'reddit_sentiment': reddit_sentiment_model.config.num_labels*3,
            'twitter_sector': twitter_sector_model.config.num_labels
        }
        self.mappings = {
            'twitter_sentiment': twitter_sentiment_model.config.id2label,
            'reddit_sentiment': reddit_sentiment_model.config.id2label,
            'twitter_sector': twitter_sector_model.config.id2label
        }

    import numpy as np

    def forward(self, tweet, reddit_body, positive_comments, negative_comments):
        '''
        Forward pass for the DUNES model.
        Args:
            tweet: previous tweet
            reddit_body: previous Reddit post
            positive_comments: comments with positive sentiment
            negative_comments: comments with negative sentiment
        Returns:
            sentiment: sentiment of the current tweet (six emotion scores)
            sector: sector of the current tweet
        '''
        # Define neutral sentiment score for each emotion
        neutral_sentiment = [1/len(self.mappings['reddit_sentiment'])] * len(self.mappings['reddit_sentiment'])  # Assuming a neutral sentiment score of 0.5 for each emotion
        
        # Replace NaN values with neutral sentiment scores for each emotion
        # if pd.isna(tweet):
        #     tweet_sentiment = neutral_sentiment 
        if pd.isna(reddit_body) :
            reddit_body_sentiment = neutral_sentiment 
        else:
            reddit_body_tokens = self.reddit_sentiment_tokenizer(reddit_body, return_tensors='pt', truncation=True)
            input_ids = reddit_body_tokens['input_ids']
            # max_length = 512
            # if len(input_ids) > max_length:
            #     reddit_body_tokens['input_ids'] = input_ids[:max_length]


            reddit_body_sentiment = self.reddit_sentiment_model(**reddit_body_tokens)[0][0].detach().numpy()
        
        if pd.isna(positive_comments):
            positive_comments_sentiment = neutral_sentiment
        else:
            positive_comments_tokens = self.reddit_sentiment_tokenizer(positive_comments, return_tensors='pt', truncation=True)
            # print(positive_comments_tokens)
            # Truncate the input_ids to a maximum length of 512 tokens
            # input_ids = positive_comments_tokens['input_ids']
            # max_length = 512
            # if len(input_ids) > max_length:
            #     positive_comments_tokens['input_ids'] = input_ids[:max_length]

            positive_comments_sentiment = self.reddit_sentiment_model(**positive_comments_tokens)[0][0].detach().numpy()
        
            
        if pd.isna(negative_comments) :
            negative_comments_sentiment = neutral_sentiment
        else:
            negative_comments_tokens = self.reddit_sentiment_tokenizer(negative_comments, return_tensors='pt', truncation=True)

            # Truncate the input_ids to a maximum length of 512 tokens for negative comments
            max_length = 512

            # Negative comments
            # input_ids = negative_comments_tokens['input_ids']
            # if len(input_ids) > max_length:
            #     negative_comments_tokens['input_ids'] = input_ids[:max_length]

            negative_comments_sentiment = self.reddit_sentiment_model(**negative_comments_tokens)[0][0].detach().numpy()

        # Get the embeddings
        tweet_embedding = self.twitter_embedding_model.encode([tweet])
        # Get the sentiment for each emotion
        tweet_tokens = self.twitter_sentiment_tokenizer(tweet, return_tensors='pt')
        tweet_sentiment = self.twitter_sentiment_model(**tweet_tokens)[0][0].detach().numpy()

        
        
        
        # Get the sector
        prev_sector_tokens = self.twitter_sector_tokenizer(tweet, return_tensors='pt')
        tweet_sector = self.twitter_sector_model(**prev_sector_tokens)[0][0].detach().numpy()


        return tweet_embedding, tweet_sentiment, reddit_body_sentiment, positive_comments_sentiment, negative_comments_sentiment, tweet_sector