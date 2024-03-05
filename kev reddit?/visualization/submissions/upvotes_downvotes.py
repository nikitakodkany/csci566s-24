import pandas as pd

# Assuming your CSV data is saved in a file named 'elonmusk_posts.csv'
filename = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/reddit_data_submissions.csv'
df = pd.read_csv(filename)

# Convert 'created_utc' to a readable date format
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
# print(df)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['ups'], df['downs'], alpha=0.5)
plt.title('Upvotes vs. Downvotes')
plt.xlabel('Upvotes')
plt.ylabel('Downvotes')
plt.grid(True)
plt.show()
