import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')  # Update with your dataset path
    df[['ups', 'downs', 'score', 'controversiality']] = df[['ups', 'downs', 'score', 'controversiality']].fillna(0)
    corr_matrix = df[['ups', 'downs', 'score', 'controversiality']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Engagement Metrics and Controversiality')
    plt.show()

if __name__ == "__main__":
    main()
