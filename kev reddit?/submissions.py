import json
import csv
from tqdm import tqdm

# Define the path to the JSON file and the CSV file
json_file = 'elonmusk_submissions'
csv_file = 'reddit_data_submissions.csv'

# Get the total number of lines in the JSON file for progress tracking
total_lines = sum(1 for line in open(json_file))

# Open the JSON file for reading and the CSV file for writing
with open(json_file, 'r') as f, open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Define fieldnames for the CSV file
    fieldnames = ['title', 'url', 'score', 'ups', 'downs', 'num_comments', 'author', 'created_utc', 'subreddit']
    
    # Create a CSV writer object
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    
    # Write header to the CSV file
    writer.writeheader()
    
    # Initialize tqdm with the total number of lines
    progress_bar = tqdm(total=total_lines, desc="Processing JSON lines", unit="line")
    
    # Read each line from the JSON file and write individual data points to the CSV file
    for line in f:
        # Increment the progress bar
        progress_bar.update(1)
        
        # Load JSON from the line
        json_data = json.loads(line)
        
        # Extract relevant fields
        data_to_write = {key: json_data.get(key, '') for key in fieldnames}
        
        # Write data to the CSV file
        writer.writerow(data_to_write)
    
    # Close the progress bar
    progress_bar.close()
