import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/output.csv')

# Count total entries
total_entries = len(df)

# Count empty rows in each category
empty_rows = df.isnull().sum()

print("Total entries:", total_entries)
print("\nEmpty rows in each category:")
print(empty_rows)
