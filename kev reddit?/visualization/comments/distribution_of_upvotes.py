import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a DataFrame
df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')

# Distribution of upvotes
plt.figure(figsize=(10, 6))
sns.histplot(df['ups'], kde=True)
plt.title('Distribution of Upvotes')
plt.xlabel('Upvotes')
plt.ylabel('Frequency')
plt.show()
