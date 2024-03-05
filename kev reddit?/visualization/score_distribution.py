import pandas as pd

# Assuming your CSV data is saved in a file named 'elonmusk_posts.csv'
filename = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/reddit_data_submissions.csv'
df = pd.read_csv(filename)

# Convert 'created_utc' to a readable date format
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
# print(df)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['score'], bins=5000, color='skyblue', edgecolor='black')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()