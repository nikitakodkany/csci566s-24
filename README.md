## CSCI 566 Spring 24 - Group Project

### Understanding

#### Data to be Collected
- 1. tweet content (t-1)
- 2. tweet content (t)
- 3. reddit content (t-1) 
- 4. #likes (t-1)
- 5. #retweets (t-1)
- 6. #comments (t-1)
- 7. timestamp (t-1)
- 8. timestamp (t)

#### Features for Prediction: [WORK DONE]
- **Embeddings (Prep)**: Text embeddings for the tweet and Reddit content. 
- **Sentiment Prediction**: Predicted sentiment (positive, negative, neutral) for the tweet and Reddit content.
- **Sector Prediction**: Predicted sector (e.g., technology, finance) for the tweet.
- **Reddit Sentiment (Aggregate)**: Aggregated sentiment score derived from the sentiment predictions of all textual content pushed through sentiment prediction model.
- 1 2 -> embeddings (prep) , sentiment pred, sector pred
- 3 -> reddit sentiment (this has to be an aggregate of all textual content pushed through to pred sentiment)

#### Input Data:
- **Tweet embedding (t-1)**: Text content of the previous tweet. [[1024, 1D]]
- **Tweet embedding (t)**: Text content of the current tweet. [[1024, 1D]]
- **Tweet sentiment (t)** : [[X , X, X]]
- **Tweet sentiment (t-1)** : [[X , X, X]]
- **Reddit embedding (t-1)**: Text content from Reddit related to the previous tweet.
- **Sector (t)**: [[20]]
-  **Sector (t-1)**: [[20]]
- ⚠️ **#Likes (t-1)**: Number of likes for the prev tweet. TO EXTRACT // NORMALIZE
- ⚠️ **#Retweets (t-1)**: Number of retweets for the prev tweet. TO EXTRACT // NORMALIZE
- ⚠️ **#Comments (t-1)**: Number of comments for the prev tweet. TO EXTRACT // NORMALIZE
- ⚠️ **Timestamp (t-1)**: Timestamp of the previous tweet. TO EXTRACT 
- ⚠️ **Timestamp (t)**: Timestamp of the current tweet. TO EXTRACT
