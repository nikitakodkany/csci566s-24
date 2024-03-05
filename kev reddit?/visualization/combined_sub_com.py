import csv

# Load submissions data
submissions = {}
with open('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_submissions.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        submissions[row['id']] = row

# Load comments data
comments_with_submissions = []
with open('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        submission_id = row['parent_id'].split('_')[1]  # Extract submission ID from parent_id
        if submission_id in submissions:
            comment_with_submission = {
                'submission': submissions[submission_id],
                'comment': row
            }
            comments_with_submissions.append(comment_with_submission)

# Write matched submission-comment pairs to a new CSV file
with open('matched_submission_comment_pairs.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['submission_id', 'submission_title', 'submission_url', 'submission_score', 'submission_ups', 'submission_downs', 'submission_num_comments', 'submission_subreddit', 'comment_author', 'comment_body', 'comment_ups', 'comment_downs', 'comment_score', 'comment_controversiality', 'submission_created_utc', 'comment_created_utc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for pair in comments_with_submissions:
        writer.writerow({
            'submission_id': pair['submission']['id'],
            'submission_title': pair['submission']['title'],
            'submission_url': pair['submission']['url'],
            'submission_score': pair['submission']['score'],
            'submission_ups': pair['submission']['ups'],
            'submission_downs': pair['submission']['downs'],
            'submission_num_comments': pair['submission']['num_comments'],
            'submission_subreddit': pair['submission']['subreddit'],
            'comment_author': pair['comment']['author'],
            'comment_body': pair['comment']['body'],
            'comment_ups': pair['comment']['ups'],
            'comment_downs': pair['comment']['downs'],
            'comment_score': pair['comment']['score'],
            'comment_controversiality': pair['comment']['controversiality'],
            'submission_created_utc': pair['submission']['created_utc'],
            'comment_created_utc': pair['comment']['created_utc']
        })
