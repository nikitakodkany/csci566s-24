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