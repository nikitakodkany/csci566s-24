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

    def forward(self, tweet, reddit_body, positive_comments, negative_comments):
        '''
        Forward pass for the DUNES model.
        Args:
            tweet: previous tweet
            curr_tweet: current tweet
            reddit_body: previous Reddit post
            curr_reddit: current Reddit post
        Returns:
            sentiment: sentiment of the current tweet
            sector: sector of the current tweet
        '''
        # Get the embeddings
        tweet_embedding = self.twitter_embedding_model.encode([tweet])

        # Get the sentiment
        tweet_tokens = self.twitter_sentiment_tokenizer(tweet, return_tensors='pt')
        tweet_sentiment = self.twitter_sentiment_model(**tweet_tokens)[0][0].detach().numpy()

        reddit_body_tokens = self.reddit_sentiment_tokenizer(reddit_body, return_tensors='pt')
        reddit_body_sentiment = self.reddit_sentiment_model(**reddit_body_tokens)[0][0].detach().numpy()
        
        positive_comments_tokens = self.reddit_sentiment_tokenizer(positive_comments, return_tensors='pt')
        positive_comments_sentiment = self.reddit_sentiment_model(**positive_comments_tokens)[0][0].detach().numpy()
        
        negative_comments_tokens = self.reddit_sentiment_tokenizer(negative_comments, return_tensors='pt')
        negative_comments_sentiment = self.reddit_sentiment_model(**negative_comments_tokens)[0][0].detach().numpy()

        # Get the sector
        prev_sector_tokens = self.twitter_sector_tokenizer(tweet, return_tensors='pt')
        tweet_sector = self.twitter_sector_model(**prev_sector_tokens)[0][0].detach().numpy()


        return tweet_embedding, tweet_sentiment, reddit_body_sentiment, positive_comments_sentiment, negative_comments_sentiment, tweet_sector