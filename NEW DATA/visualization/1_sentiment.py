import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load and prepare the dataset
df = pd.read_csv('elonmusk/NEW DATA/elonmusk.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plotting sentiment over time
plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['Sentiment'], marker='o', linestyle='', ms=2)
plt.title('Sentiment of Tweets Over Time')
plt.xlabel('Time')
plt.ylabel('Sentiment')
plt.show()
