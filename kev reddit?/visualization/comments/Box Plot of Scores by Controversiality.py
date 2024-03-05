import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')

plt.figure(figsize=(10, 6))
sns.boxplot(x='controversiality', y='score', data=df)
plt.title('Box Plot of Scores by Controversiality')
plt.xlabel('Controversiality')
plt.ylabel('Score')
plt.show()
