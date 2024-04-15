import numpy as np
import torch

def reformat_dataset(data):
    """
    Reformat the dataset to be compatible with the DataLoaderDUNES class
    """
    reformatted_data = []
    for index, row in data.iterrows():
        reformatted_data.append({
            'tweet_content': row['tweet_content'],
            'title_submission': row['title_submission'],
            'positive_comments': row['positive_comments'],
            'negative_comments': row['negative_comments'],
            'score_submission': row['score_submission'],
            'positive_scores': row['positive_scores'],
            'negative_scores': row['negative_scores'],
            'num_likes': row['num_likes'],
            'num_retweets': row['num_retweets'],
            'num_replies': row['num_replies']
        })
    
    return reformatted_data

def print_target_stats(data):
    """
    Print the target mean, standard deviation, max, and min in table format
    """
    target_stats = data[['num_likes', 'num_retweets', 'num_replies']].describe().loc[['mean', 'std', 'max', 'min']]
    print(target_stats)

def calculate_mean_absolute_error(predictions, targets):
    """
    Calculate the mean absolute error for the target variables
    """
    return np.mean(np.abs(predictions - targets))

def quantile_loss(preds, targets, quantile=0.5):
    assert 0 < quantile < 1, "Quantile should be in (0, 1)"
    errors = targets - preds
    return torch.max((quantile - 1) * errors, quantile * errors).mean()