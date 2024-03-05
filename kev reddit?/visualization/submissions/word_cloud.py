import pandas as pd

# Assuming your CSV data is saved in a file named 'elonmusk_posts.csv'
filename = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/reddit_data_submissions.csv'
df = pd.read_csv(filename)

# Convert 'created_utc' to a readable date format
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Join all titles into a single string
all_titles = ' '.join(df['title'])

# Generate a word cloud image
wordcloud = WordCloud(background_color='white', width=800, height=400, max_font_size=100).generate(all_titles)

# Display the generated image:
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
