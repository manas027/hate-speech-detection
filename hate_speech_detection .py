import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import classify
from nltk import NaiveBayesClassifier
import re
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_tweet(tweet):
    # Remove URLs, usernames, and hashtags
    tweet = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    tweet = tweet.lower()
    # Tokenize the tweet
    tokens = word_tokenize(tweet)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in tweet_words)
    return features

# Load training data
positive_tweets = [('I love this place', 'positive'),
                   ('This food is amazing', 'positive'),
                   ('Great service!', 'positive'),
                   ('I hate this restaurant', 'negative'),
                   ('The food was terrible', 'negative'),
                   ('Awful experience', 'negative')]
negative_tweets = [('This movie is so bad', 'negative'),
                   ('I can\'t stand this song', 'negative'),
                   ('Terrible customer service', 'negative'),
                   ('I enjoyed the concert', 'positive'),
                   ('Amazing performance!', 'positive'),
                   ('Loved the show', 'positive')]

tweets = []
for (words, sentiment) in positive_tweets + negative_tweets:
    processed_words = preprocess_tweet(words)
    tweets.append((processed_words, sentiment))

# Extract features
all_words = []
for (words, sentiment) in tweets:
    all_words.extend(words)

word_freq = nltk.FreqDist(all_words)
word_features = list(word_freq.keys())[:100]

training_data = nltk.classify.apply_features(extract_features, tweets)

# Train the classifier
classifier = NaiveBayesClassifier.train(training_data)

# Test the classifier
test_tweet = "terrible customer service"
processed_test_tweet = preprocess_tweet(test_tweet)
test_features = extract_features(processed_test_tweet)
result = classifier.classify(test_features)
print("Test Tweet: {}".format(test_tweet))
print("Sentiment: {}".format(result))

