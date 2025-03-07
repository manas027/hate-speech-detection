import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
import re
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function to cleaning and preparation of tweets
def preprocess_tweet(tweet):
    """
    Cleans the input tweet by:
    - Removing URLs, usernames, and hashtags
    - Removing punctuation
    - Converting to lowercase
    - Tokenizing the text
    - Removing stopwords
    - Lemmatizing the words

    Parameters:
        tweet (str): The raw tweet text.

    Returns:
        list: A list of cleaned and lemmatized tokens.
    """
    # Remove links, usernames, and hashtags
    tweet = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", tweet)
    
    # Remove punctuation and convert to lowercase
    tweet = tweet.translate(str.maketrans("", "", string.punctuation)).lower()
    
    # Tokenize the text into individual words
    tokens = word_tokenize(tweet)
    
    # Remove common stopwords like "the", "is", etc.
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize tokens to their base form
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens

# Function to extract features for the classifier
def extract_features(tweet_tokens):
    """
    Converts a list of tokens into a dictionary of features.
    
    Parameters:
        tweet_tokens (list): A list of processed tokens.

    Returns:
        dict: A dictionary of features for classification.
    """
    tweet_words = set(tweet_tokens)
    features = {}
    for word in word_features:
        # Mark if the word is present in the tweet
        features[f'contains({word})'] = (word in tweet_words)
    return features

# Sample training data with labeled sentiments
positive_tweets = [
    ('I love this place', 'positive'),
    ('This food is amazing', 'positive'),
    ('Great service!', 'positive')
]

negative_tweets = [
    ('I hate this restaurant', 'negative'),
    ('The food was terrible', 'negative'),
    ('Awful experience', 'negative')
]

# Combining positive and negative tweets
all_tweets = positive_tweets + negative_tweets

# Preprocess each tweet and store it along with its labels
processed_tweets = []
for (tweet, sentiment) in all_tweets:
    processed_tweet = preprocess_tweet(tweet)
    processed_tweets.append((processed_tweet, sentiment))

# Build a list of all words across tweets for feature extraction
all_words = []
for (tokens, sentiment) in processed_tweets:
    all_words.extend(tokens)

# Frequency distribution of words
word_freq = nltk.FreqDist(all_words)

#top 100 words as features
word_features = list(word_freq.keys())[:100]

#feature sets for training
training_data = [(extract_features(tokens), sentiment) for (tokens, sentiment) in processed_tweets]

# Training of naive bayes Classifier
classifier = NaiveBayesClassifier.train(training_data)

# Testing the classifier with sample tweet
def test_classifier(tweet):
    """
    Tests the sentiment of a given tweet.

    Parameters:
        tweet (str): The tweet text to classify.

    Returns:
        str: The predicted sentiment ('positive' or 'negative').
    """
    processed_tweet = preprocess_tweet(tweet)
    features = extract_features(processed_tweet)
    sentiment = classifier.classify(features)
    return sentiment

# Run the test
if __name__ == "__main__":
    # Sample tweets for testing
    test_tweets = [
        "The customer service was amazing!",
        "I had a terrible experience at the restaurant.",
        "Loved the food and the atmosphere.",
        "The service was awful, but the food was okay."
    ]

    print("Sentiment Analysis Results:")
    for tweet in test_tweets:
        sentiment = test_classifier(tweet)
        print(f"Tweet: {tweet}\nPredicted Sentiment: {sentiment}\n")

    # Display the most informative features
    print("\nMost Informative Features:")
    classifier.show_most_informative_features(5)


