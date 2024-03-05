import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_submissions.csv')

# Sort and display the top 10 for each column
for column in ['score', 'ups', 'downs', 'num_comments']:
    sorted_df = df.sort_values(by=column, ascending=False)
    top_10 = sorted_df.head(10)
    print(f"Top 10 entries sorted by {column}:")
    print(top_10)
    print("\n")


import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_submissions.csv')

# Sort the DataFrame by multiple columns
sorted_df = df.sort_values(by=['score', 'ups', 'downs', 'num_comments'], ascending=False)

# Display the top 10 rows
top_10 = sorted_df.head(10)
print(top_10)

