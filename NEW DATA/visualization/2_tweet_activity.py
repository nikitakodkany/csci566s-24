import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

df = pd.read_csv('elonmusk/NEW DATA/elonmusk.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

df.set_index('Timestamp', inplace=True)

df.resample('M').size().plot(figsize=(10, 6))
plt.title('Tweet Activity Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Tweets')
plt.show()
