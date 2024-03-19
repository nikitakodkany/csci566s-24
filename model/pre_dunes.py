import torch
import torch.nn as nn

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

    def forward(self, prev_tweet, curr_tweet, prev_reddit):
        '''
        Forward pass for the DUNES model.
        Args:
            prev_tweet: previous tweet
            curr_tweet: current tweet
            prev_reddit: previous Reddit post
            curr_reddit: current Reddit post
        Returns:
            sentiment: sentiment of the current tweet
            sector: sector of the current tweet
        '''
        # Get the embeddings
        prev_tweet_embedding = self.twitter_embedding_model.encode([prev_tweet])
        curr_tweet_embedding = self.twitter_embedding_model.encode([curr_tweet])

        # Get the sentiment
        prev_tweet_tokens = self.twitter_sentiment_tokenizer(prev_tweet, return_tensors='pt')
        prev_tweet_sentiment = self.twitter_sentiment_model(**prev_tweet_tokens)[0][0].detach().numpy()
        curr_tweet_tokens = self.twitter_sentiment_tokenizer(curr_tweet, return_tensors='pt')
        curr_tweet_sentiment = self.twitter_sentiment_model(**curr_tweet_tokens)[0][0].detach().numpy()

        prev_reddit_tokens = self.reddit_sentiment_tokenizer(prev_reddit, return_tensors='pt')
        prev_reddit_sentiment = self.reddit_sentiment_model(**prev_reddit_tokens)[0][0].detach().numpy()

        # Get the sector
        prev_sector_tokens = self.twitter_sector_tokenizer(prev_tweet, return_tensors='pt')
        prev_tweet_sector = self.twitter_sector_model(**prev_sector_tokens)[0][0].detach().numpy()
        curr_sector_tokens = self.twitter_sector_tokenizer(curr_tweet, return_tensors='pt')
        curr_tweet_sector = self.twitter_sector_model(**curr_sector_tokens)[0][0].detach().numpy()


        return prev_tweet_embedding, curr_tweet_embedding, prev_tweet_sentiment, curr_tweet_sentiment, prev_reddit_sentiment, prev_tweet_sector, curr_tweet_sector
        