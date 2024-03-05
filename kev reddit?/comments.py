import json
import csv

# The path to your input text file
input_file_path = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk_comments'

# The path to your output CSV file
output_file_path = 'reddit_data_comments.csv'

# Define the headers for the CSV file based on the features you want to extract
headers = ['author', 'body', 'created_utc', 'subreddit', 'ups', 'downs', 'score', 'controversiality']

with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(headers)  # Write the headers to the CSV file

    for line in infile:
        # Parse the JSON object from each line of the file
        data = json.loads(line)
        
        # Clean up the 'body' field to remove newline characters and other potential issues
        if 'body' in data:
            data['body'] = data['body'].replace('\n', ' ').replace('\r', ' ')
        
        # Extract the data you're interested in
        row = [data.get(header, '') for header in headers]
        
        # Write the extracted data to the CSV file
        csv_writer.writerow(row)

print("Data extraction and CSV file creation completed successfully.")
