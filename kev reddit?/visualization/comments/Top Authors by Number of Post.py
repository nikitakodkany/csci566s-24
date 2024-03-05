from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')
# Count the number of posts by each author
top_authors = df['author'].value_counts().head(10)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x=top_authors.values, y=top_authors.index)
plt.title('Top Authors by Number of Posts')
plt.xlabel('Number of Posts')
plt.ylabel('Author')
plt.show()
