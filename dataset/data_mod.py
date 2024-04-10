import pandas as pd

# Read the CSV file
df = pd.read_csv('elon_twitter_data.csv')

# Filter the rows where title_submission is not nan
filtered_df = df[df['title_submission'].notnull()]

# Write the filtered data to a new CSV file
filtered_df.to_csv('elon_reddit_data.csv', index=False)