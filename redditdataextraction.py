import praw
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reddit API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'
user_agent = 'your_user_agent'

# Initialize Reddit API wrapper
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

def extract_reddit_data(tweet_text, num_comments=100):
    # Extract Reddit comments related to a particular tweet text
    reddit_comments = []
    for comment in reddit.search(f'selftext:"{tweet_text}"'):
        reddit_comments.append(comment.body)
        if len(reddit_comments) >= num_comments:
            break
    return reddit_comments

def calculate_aggregate_sentiment(reddit_comments):
    # Calculate sentiment polarity for each comment and aggregate the scores
    total_polarity = sum(TextBlob(comment).sentiment.polarity for comment in reddit_comments)
    # Calculate the aggregate sentiment score as the average polarity of all comments
    aggregate_sentiment_score = total_polarity / len(reddit_comments)
    return aggregate_sentiment_score

def calculate_tfidf_correlation(tweet_text, reddit_comments):
    # Perform TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    all_text = [tweet_text] + reddit_comments
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    # Calculate cosine similarity between the tweet and Reddit comments
    similarity_matrix = cosine_similarity(tfidf_matrix)
    tweet_index = 0  # Index of the tweet in the similarity matrix
    correlations = similarity_matrix[tweet_index][1:]  # Exclude the similarity with the tweet itself
    return correlations

def main():
    # Example tweet text
    tweet_text = 'Your tweet content goes here'

    # Extract Reddit comments related to the tweet text
    reddit_comments = extract_reddit_data(tweet_text)

    # Calculate aggregate sentiment score
    aggregate_sentiment_score = calculate_aggregate_sentiment(reddit_comments)
    print("Aggregate Sentiment Score:", aggregate_sentiment_score)

    # Calculate TF-IDF correlation
    correlations = calculate_tfidf_correlation(tweet_text, reddit_comments)
    print('Correlation between the tweet and Reddit comments:')
    for i, correlation in enumerate(correlations):
        print(f'Reddit Comment {i+1}: {correlation}')

if __name__ == '__main__':
    main()
