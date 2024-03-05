import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

def main():
    # Load the data
    df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')
    
    # Convert 'created_utc' to numeric, handling errors, then to datetime
    df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
    df['created_date'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    
    # Drop rows with NaT in 'created_date'
    df = df.dropna(subset=['created_date'])
    
    # Diagnostic print
    print(f"Rows after dropping NaT in 'created_date': {len(df)}")

    # Fill missing or empty bodies with a placeholder
    df['body'] = df['body'].fillna('No content')

    # Compute sentiment scores
    df['sentiment'] = df['body'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Set 'created_date' as the DataFrame index for resampling
    df.set_index('created_date', inplace=True)

    # Diagnostic print for sentiment
    print(f"Sentiment column after computation: {df['sentiment'].describe()}")

    # Resample and plot sentiment over time
    df['sentiment'].resample('D').mean().plot(kind='line', figsize=(10, 6))
    plt.title('Average Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
