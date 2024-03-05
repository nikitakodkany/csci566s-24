from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')

nltk.download('stopwords')

# Load your data into df
# df = pd.read_csv('your_file_path.csv')

# Get English stopwords and update with any subreddit-specific stopwords
stop_words = set(stopwords.words('english'))
# Convert the set to a list
stop_words_list = list(stop_words)
# Add subreddit-specific stopwords if necessary
# stop_words_list.extend(['elon', 'musk', 'tesla', 'spacex'])
stop_words_list.extend(['removed', 'elon', 'people', 'like', 'would', 'musk'])

# Instantiate a CountVectorizer, using the list of stopwords
vectorizer = CountVectorizer(stop_words=stop_words_list)

# Fit and transform the comments
X = vectorizer.fit_transform(df['body'].astype(str))

# Sum word occurrences
word_counts = X.sum(axis=0)
words_freq = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Convert to DataFrame for easy plotting
words_df = pd.DataFrame(words_freq, columns=['word', 'frequency'])

# Plot the top 20 words
plt.figure(figsize=(12, 8))
sns.barplot(x='frequency', y='word', data=words_df.head(20), palette='viridis')
plt.title('Top 20 Words in Comments (Excluding Common Stopwords)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()
