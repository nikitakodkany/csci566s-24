import pandas as pd

# Assuming your CSV data is saved in a file named 'elonmusk_posts.csv'
filename = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/reddit_data_submissions.csv'
df = pd.read_csv(filename)

# Convert 'created_utc' to a readable date format
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
# print(df)
import matplotlib.pyplot as plt

top_contributors = df['author'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_contributors.plot(kind='bar')
plt.title('Top 10 Contributors')
plt.xlabel('Author')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45)
plt.show()
