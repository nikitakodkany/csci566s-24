import pandas as pd

def inspect_created_utc():
    df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')
    print(df['created_utc'].head())  # Print the first few values of 'created_utc'
    print(df['created_utc'].describe())  # Get a description of the column

if __name__ == "__main__":
    inspect_created_utc()
