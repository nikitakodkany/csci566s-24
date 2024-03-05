import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')
# Convert 'created_utc' to datetime
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

# Resample to get the count of posts per month
posts_per_month = df.resample('M', on='created_utc').size()

# Plotting
plt.figure(figsize=(12, 6))
posts_per_month.plot()
plt.title('Number of Posts Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Posts')
plt.show()