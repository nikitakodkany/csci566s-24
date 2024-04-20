import re
import numpy as np
import pandas as pd

df = pd.read_csv(r"dataset/Elon_Tweets_DataFinal.csv")

def remove_special_characters(text):
    text = str(text)
    if text == 'nan':
        return ''
    return text.encode('ascii', 'ignore').decode()

# Function to process comments and categorize them
def process_comments(comment_str):
    positive_comments = []
    negative_comments = []
    positive_scores = 0
    negative_scores = 0
    
    # Regular expression pattern to extract comment objects
    pattern_pos = r"{'score_comment': (\d+), 'created_utc_comment': \d+, 'body': \"(.*?)\"}|{'score_comment': (\d+), 'created_utc_comment': \d+, 'body': '(.*?)'}"
    pattern_neg = r"{'score_comment': -(\d+), 'created_utc_comment': \d+, 'body': \"(.*?)\"}|{'score_comment': -(\d+), 'created_utc_comment': \d+, 'body': '(.*?)'}"
    
    # Find all matches of comment objects in the string
    matches_pos = re.findall(pattern_pos, comment_str)
    matches_neg = re.findall(pattern_neg, comment_str)
    
    for match in matches_pos:
        if match[0]:
            score = int(match[0])
            body = match[1]
        else:
            score = int(match[2])
            body = match[3]

        if score == 0:
            negative_comments.append(body.strip())
        else:
            positive_comments.append(body.strip())
            positive_scores += score
    
    for match in matches_neg:
        if match[0]:
            score = int(match[0])
            body = match[1]
        else:
            score = int(match[2])
            body = match[3]
        
        negative_comments.append(body.strip())
        negative_scores += score
    
    return {
        'positive_comments': ' '.join(positive_comments),
        'positive_scores': positive_scores,
        'negative_comments': ' '.join(negative_comments),
        'negative_scores': negative_scores
    }

def combine(df):
    df['body_submission'] = df['body_submission'].fillna('') + ' ' + df['comments'].fillna('')
    df[['positive_comments', 'positive_scores', 'negative_comments', 'negative_scores']] = df['comments'].apply(lambda x: pd.Series(process_comments(x)) if x is not np.nan else pd.Series({'positive_comments': '', 'positive_scores': '', 'negative_comments': '', 'negative_scores': ''}))
    df.rename(columns={'new_likes': 'num_likes', 'new_retweets': 'num_retweets', 'new_replies': 'num_replies'}, inplace=True)
    df.drop(['body_submission', 'comments'], axis=1, inplace=True)

    columns_to_clean = ['tweet_content', 'title_submission', 'positive_comments', 'negative_comments']
    df[columns_to_clean] = df[columns_to_clean].applymap(remove_special_characters)

    df.to_csv(r"data/COMBINED_FINAL.csv", index=False)

combine(df)