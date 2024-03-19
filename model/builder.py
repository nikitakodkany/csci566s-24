import numpy as np
from typing import Dict
from scipy.special import expit, softmax

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from .pre_dunes import PreDUNES

def create_preprocessing_model(
        twitter_embedding: str,
        twitter_sentiment: str,
        reddit_sentiment: str,
        twitter_sector: str
):
    '''
    Initialize a DUNES model from a set of embeddings and sentiment models.
    Feeds the output of these models to a transformer for classification.
    Args:
        twitter_embedding: path to the Twitter embedding huggingface model
        twitter_sentiment: path to the Twitter sentiment huggingface model
        reddit_sentiment: path to the Reddit sentiment huggingface model
        twitter_sector: path to the Twitter sector huggingface model
    '''

    # Load the models
    twitter_embedding_model = SentenceTransformer(twitter_embedding)
    twitter_sentiment_tokenizer = AutoTokenizer.from_pretrained(twitter_sentiment)
    twitter_sentiment_model = AutoModelForSequenceClassification.from_pretrained(twitter_sentiment)
    reddit_sentiment_tokenizer = AutoTokenizer.from_pretrained(reddit_sentiment)
    reddit_sentiment_model = AutoModelForSequenceClassification.from_pretrained(reddit_sentiment)
    twitter_sector_tokenizer = AutoTokenizer.from_pretrained(twitter_sector)
    twitter_sector_model = AutoModelForSequenceClassification.from_pretrained(twitter_sector)

    # Freeze the models
    twitter_sentiment_model.eval()
    twitter_sentiment_model.requires_grad_(False)
    reddit_sentiment_model.eval()
    reddit_sentiment_model.requires_grad_(False)
    twitter_sector_model.eval()
    twitter_sector_model.requires_grad_(False)
    
    # Create the DUNES model
    model = PreDUNES(
        twitter_embedding_model, 
        twitter_sentiment_tokenizer, 
        twitter_sentiment_model, 
        reddit_sentiment_tokenizer, 
        reddit_sentiment_model, 
        twitter_sector_tokenizer, 
        twitter_sector_model
        )

    return model

def printClassMappings(model, predictions):
    predictions = softmax(predictions)
    class_mapping = AutoModel.from_pretrained(model).config.id2label

    ranking = np.argsort(predictions)
    ranking = ranking[::-1]
    for i in range(predictions.shape[0]):
        l = class_mapping[ranking[i]]
        s = predictions[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")

def test():
    preprocessing_model = create_preprocessing_model(
        "mixedbread-ai/mxbai-embed-large-v1",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "SamLowe/roberta-base-go_emotions",
        "cardiffnlp/tweet-topic-latest-multi"
    )

    prev_tweet_embedding, curr_tweet_embedding, prev_tweet_sentiment, curr_tweet_sentiment, prev_reddit_sentiment, prev_tweet_sector, curr_tweet_sector = preprocessing_model(
        "@WholeMarsBlog Headline is misleading. Starlink can obviously offer far more robust positioning than GPS, as it will have ~1000X more satellites over time. Not all will have line of sight to users, but still &gt;10X GPS &amp; far stronger signal. Just not today’s problem.",
        "@spideycyp_155 @BillyM2k If Russia faced calamitous defeat in conventional warfare for something as strategically critical as Crimea, the probability of using nuclear weapons is high",
        "We know who controls the media. The same corporations who have wreaked havoc on the globe for decades, if not centuries, the big banks who financed them, and the governments who turned a blind eye to the destruction. The same entities who have brought us to the precipice of destruction - quite possibly condemning us, and our progeny to an unlivable climate They have tried to stop you at every turn, and yet you persist for the good of humanity. We love you, Elon! Keep up the good work! As you have said, we must never let the light of human consciousness fade - never!"
    )

    print("prev_tweet_embedding:", prev_tweet_embedding)
    print("curr_tweet_embedding:", curr_tweet_embedding)
    print("prev_tweet_sentiment:", prev_tweet_sentiment)
    print("curr_tweet_sentiment:", curr_tweet_sentiment)

    print("prev_reddit_sentiment:")
    printClassMappings("SamLowe/roberta-base-go_emotions", prev_reddit_sentiment)
    print("prev_tweet_sector:")
    printClassMappings("cardiffnlp/tweet-topic-latest-multi", prev_tweet_sector)
    print("curr_tweet_sector:")
    printClassMappings("cardiffnlp/tweet-topic-latest-multi", curr_tweet_sector)
