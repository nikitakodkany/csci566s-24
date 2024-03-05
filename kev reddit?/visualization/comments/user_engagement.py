import pandas as pd

# Load the data
data = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')

# Assuming 'author' column contains the username and 'ups' column contains the upvotes

# Calculate total upvotes for each user
user_upvotes = data.groupby('author')['ups'].sum().reset_index()

# Calculate total comments for each user
user_comments = data['author'].value_counts().reset_index()
user_comments.columns = ['author', 'comment_count']

# Merge the two dataframes
user_engagement = pd.merge(user_upvotes, user_comments, on='author')

# Calculate engagement score (you can define this based on your requirements)
user_engagement['engagement_score'] = user_engagement['ups'] * user_engagement['comment_count']

# Sort users by engagement score
user_engagement = user_engagement.sort_values(by='engagement_score', ascending=False)

# Display the top engaged users
print(user_engagement.head(10))
