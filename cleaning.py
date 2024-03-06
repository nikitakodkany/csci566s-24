import csv

# Function to check if all values in a row are present
def is_complete(row):
    return all(value.strip() for value in row)

# Open the input CSV file
with open('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/output.csv', 'r', newline='') as infile:
    reader = csv.reader(infile)
    next(reader)  # Skip header row

    # Filter out rows with absent values
    complete_rows = [row for row in reader if is_complete(row)]

# Open a new CSV file to write the filtered data
with open('filtered.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    # Write the header row
    writer.writerow(['likes_count', 'tweets', 'retweets_count', 'date', 'mentions', 'replies_count', 'comments_count'])

    # Write the filtered rows
    writer.writerows(complete_rows)

print("Filtered data has been written to 'output.csv'")
