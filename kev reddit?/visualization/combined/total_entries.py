import csv

# Initialize counters
total_entries = 0
empty_rows = {'submission_id': 0, 'submission_title': 0, 'submission_url': 0, 'submission_score': 0, 'submission_ups': 0, 'submission_downs': 0,
              'submission_num_comments': 0, 'submission_subreddit': 0, 'comment_author': 0, 'comment_body': 0, 'comment_ups': 0, 'comment_downs': 0,
              'comment_score': 0, 'comment_controversiality': 0, 'submission_created_utc': 0, 'comment_created_utc': 0}

# Open the CSV file
with open('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/matched_submission_comment_pairs.csv', 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        total_entries += 1  # Increment total entries counter
        
        # Check each column for emptiness and update respective empty rows counter
        for column in row:
            if not row[column].strip():
                empty_rows[column] += 1

# Display the results
print("Total entries:", total_entries)
print("\nEmpty rows in each category:")
for category, count in empty_rows.items():
    print(f"{category}: {count}")
