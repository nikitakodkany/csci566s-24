from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')
# Join all the comments into a single string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure all body entries are strings
df['body'] = df['body'].astype(str)

# Join all the comments into a single string
text = ' '.join(body for body in df['body'])

# Generate and display the word cloud
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Post Bodies')
plt.show()

