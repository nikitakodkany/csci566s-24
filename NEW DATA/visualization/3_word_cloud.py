import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load and prepare the dataset
df = pd.read_csv('elonmusk/NEW DATA/elonmusk.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# df['Sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
from wordcloud import WordCloud

# Generating the word cloud
text = ' '.join(tweet for tweet in df['text'])
wordcloud = WordCloud(background_color='white', max_words=200, contour_width=3, contour_color='steelblue').generate(text)

# Display the generated image:
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
