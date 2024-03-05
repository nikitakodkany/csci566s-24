import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

# Load the data
data = pd.read_csv('/Users/kevinsherla/AAAA/USC/CSCI DEEEEEEEEP L/STEP 1/elonmusk/kev reddit?/reddit_data_comments.csv')

# Preprocess the text data
# Assuming 'body' column contains the comment text
corpus = data['body'].fillna('').values.astype('U')

# Vectorize the text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(corpus)

# Apply LDA with progress bar
n_topics = 5  # Number of topics
lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', random_state=0)

# Initialize tqdm for progress tracking
with tqdm(total=lda_model.max_iter) as pbar:
    for i in range(lda_model.max_iter):
        lda_model.partial_fit(X)
        pbar.update(1)

# Display the top words for each topic
def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

n_top_words = 10  # Number of top words to display for each topic
feature_names = vectorizer.get_feature_names_out()
display_topics(lda_model, feature_names, n_top_words)
