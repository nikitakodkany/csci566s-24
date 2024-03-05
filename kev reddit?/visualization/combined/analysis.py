import pandas as pd
from textblob import TextBlob
from mlxtend.frequent_patterns import association_rules, apriori

# Load the data
df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/matched_submission_comment_pairs.csv')

# 1. Sentiment Analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['comment_sentiment'] = df['comment_body'].apply(get_sentiment)

# 8. Association Rules
# Tokenize submission titles and comments
df['submission_title_tokens'] = df['submission_title'].str.split()
df['comment_body_tokens'] = df['comment_body'].str.split()

# Create transactions
transactions = df['submission_title_tokens'].append(df['comment_body_tokens']).dropna()

# Perform one-hot encoding
encoded_transactions = pd.get_dummies(transactions.apply(pd.Series).stack()).sum(level=0)

# Generate frequent itemsets
frequent_itemsets = apriori(encoded_transactions, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 9. Content Analysis
# Extract key themes or topics
# For example, you can use techniques like LDA or TF-IDF to extract topics from text data

# 10. User Behavior Analysis
# Analyze the behavior of specific users
# For example, you can group the data by 'comment_author' and analyze their commenting activity
user_behavior = df.groupby('comment_author').agg({
    'submission_id': 'count',
    'comment_sentiment': 'mean',
    # Add more metrics as needed
}).reset_index()

# Print results
print("Sentiment Analysis:")
print(df[['comment_body', 'comment_sentiment']].head())

print("\nAssociation Rules:")
print(rules.head())

# Print results for Content Analysis and User Behavior Analysis according to the specific techniques used
