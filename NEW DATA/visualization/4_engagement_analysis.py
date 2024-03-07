import pandas as pd
import matplotlib.pyplot as plt

def convert_k_to_int(value):
    """
    Converts a string with K (thousands), M (millions), or commas to an integer.
    For example, '48.7K' becomes 48700, '9,005' becomes 9005, and '1.1M' becomes 1100000.
    """
    if isinstance(value, str):  # Checks if the value is a string
        value = value.replace(',', '')  # Remove commas to handle '9,005' like cases
        if 'K' in value:
            return int(float(value.replace('K', '')) * 1000)
        elif 'M' in value:
            return int(float(value.replace('M', '')) * 1000000)
        else:
            return int(value)
    return value

# Load the dataset
df = pd.read_csv('elonmusk/NEW DATA/elonmusk.csv')

# Apply the conversion to 'Likes', 'Retweets', and 'Comments' columns
df['Likes'] = df['Likes'].apply(convert_k_to_int)
df['Retweets'] = df['Retweets'].apply(convert_k_to_int)
df['Comments'] = df['Comments'].apply(convert_k_to_int)

# Calculate total engagement
df['Total Engagement'] = df['Likes'] + df['Retweets'] + df['Comments']

# Sort the DataFrame based on 'Total Engagement' to find the top 10 tweets
top_engaged_tweets = df.sort_values(by='Total Engagement', ascending=False).head(10)

# Visualization
plt.figure(figsize=(10, 8))

# Create a new column for truncated tweet text to use as labels
# Ensure that only the text is included, not any numbers or stats
top_engaged_tweets['Shortened Text'] = top_engaged_tweets['text'].str[:30] + '...'

# Plot the bar chart
plt.barh(top_engaged_tweets['Shortened Text'], top_engaged_tweets['Total Engagement'])

# Set the labels and title
plt.xlabel('Total Engagement')
plt.ylabel('Tweets')
plt.title('Top 10 Tweets by Engagement')

# Invert y-axis to have the highest engagement at the top
plt.gca().invert_yaxis()

# Show the plot
plt.show()
