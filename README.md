# Dunes of Influence: Quantitative Prediction of Influential Individuals’ Influence on Community Dynamics through Advanced Language Models

## Authors
- Abhinav Parameshwaran
- Kevin Sherla
- Mihir Pavuskar
- Muskan Pandey
- Nikita Kodkany
- Shashank Prakash

## University
Department of Computer Science, University of Southern California

## Abstract
This project develops a deep learning framework to estimate potential engagement metrics such as likes, comments, and retweets for new tweets. By leveraging data from social media platforms like Twitter and Reddit, and employing advanced natural language processing techniques, the model predicts future engagement. This project showcases the integration of multi-modal data analysis and advanced language models to enhance the understanding and prediction of content performance dynamics.

## Introduction
Social media platforms such as Twitter and Reddit play a central role in information dissemination, influencing public opinion and driving cultural trends. Predicting user engagement with content on these platforms is crucial for marketers, content creators, social scientists, and platform administrators. Traditional approaches to predicting social media engagement often fail to capture the complex interactions between content features and contextual elements.

Our project addresses these challenges by developing a multimodal predictive model leveraging advanced machine learning techniques, particularly deep learning and NLP. We focus on Elon Musk's Twitter activity as a case study due to his significant impact on social media dynamics and market movements. By integrating diverse data types, our model provides a comprehensive analysis of the factors contributing to social media engagement.

## Problem Statement
The main challenge is accurately predicting user engagement with content on social media platforms, considering the diverse and complex data sources available. Existing models often lack the ability to adapt to the dynamic nature of social media content and fail to encapsulate the multifaceted nature of user interactions.

## Solution
### Data Collection
- **Twitter Data**: 4,000 tweets from Elon Musk, excluding replies to focus on direct communications.
- **Reddit Data**: Discussions from the five most active subreddits related to Elon Musk (r/ElonMusk, r/Tesla, r/SpaceX, r/SpaceXLounge, r/EnoughMuskSpam).

### Data Pre-processing
- **Feature Extraction**: Using advanced NLP tools for semantic embeddings, sentiment analysis, and sector classification.
- **Semantic Embeddings**: Utilizing models like `mixedbread-ai/mxbai-embed-large-v1` for Twitter embeddings.
- **Sentiment Analysis**: Applying models such as `cardiffnlp/twitter-roberta-base-sentiment-latest` for Twitter and `bhadresh-savani/distilbert-base-uncased-emotion` for Reddit.
- **Sector Classification**: Using `cardiffnlp/tweet-topic-latest-multi` to classify tweets into different sectors.

### Model Architecture
- **Compression Layer**: Initial linear layers to compress high-dimensional input features.
- **Positional Encoding**: Adds temporal information to the sequence.
- **Transformer Encoder**: Processes the sequence of feature vectors.
- **Output Layer**: Maps the Transformer encoder outputs to predicted engagement metrics.

### Results and Discussion
- **Model Comparison**: Comparing the performance of different models such as MLP, LSTM, and Transformer architectures.
- **Loss Functions**: Utilizing various loss functions like MAE, MSE, Huber loss, and Quantile loss to train the model.
- **Scaling**: Employing robust scaling and logarithmic scaling to mitigate data skew.

## Future Work
1. **Data Diversity**: Integrating a broader array of data types and sources.
2. **Multimodal Analysis**: Refining the model's ability to handle various modalities (textual, audio, visual).
3. **Sentiment Nuances**: Developing more sophisticated sentiment analysis techniques.
4. **Robust Generalization**: Improving the model’s ability to generalize across diverse scenarios.

## References
A comprehensive list of references used in the project, including foundational works in social media analytics, machine learning, and NLP.

## How to Use
1. Clone the repository.
2. Install the required dependencies.
3. Prepare the dataset by following the instructions in the `data/` directory.
4. Run the pre-processing scripts to extract features.
5. Train the model using the provided training scripts.
6. Evaluate the model using the evaluation scripts to compare performance metrics.


## Contact
For any questions or suggestions, please contact kodkany@usc.edu
