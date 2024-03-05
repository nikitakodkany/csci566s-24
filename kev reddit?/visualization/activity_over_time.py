import pandas as pd

# Assuming your CSV data is saved in a file named 'elonmusk_posts.csv'
filename = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/reddit_data_submissions.csv'
df = pd.read_csv(filename)

# Convert 'created_utc' to a readable date format
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
# print(df)
import matplotlib.pyplot as plt

# Group by date and count the number of posts
posts_over_time = df.groupby(df['created_date'].dt.date).size()

# Plotting
plt.figure(figsize=(15, 6))
posts_over_time.plot(kind='line')
plt.title('Number of Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.grid(True)
plt.show()
