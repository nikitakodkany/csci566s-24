import json
import csv
from tqdm import tqdm

# Define the path to the JSON file and the CSV file
json_file = '/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk_comments'
csv_file = 'reddit_data.csv'

# Open the JSON file for reading and the CSV file for writing
with open(json_file, 'r') as f, open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Define fieldnames for the CSV file
    fieldnames = ['body', 'subreddit', 'ups', 'downs', 'score', 'author', 'controversiality', 'created_utc']
    
    # Create a CSV writer object
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    
    # Write header to the CSV file
    writer.writeheader()
    
    # Initialize tqdm to display progress bar
    progress_bar = tqdm(f, desc="Processing JSON", unit=" lines")
    
    # Read each line from the JSON file and write individual data points to the CSV file
    for line in progress_bar:
        # Load JSON from the line
        json_data = json.loads(line)
        
        # Write each data point to the CSV file
        for key in fieldnames:
            if key == 'body':
                # Enclose the body field in double quotes
                csvfile.write('"' + str(json_data.get(key, '')).replace('"', '""') + '", ')
            else:
                csvfile.write(str(json_data.get(key, '')) + ', ')
        csvfile.write('\n')
