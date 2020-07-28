from textblob import TextBlob
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import re
import tweepy
import os
import matplotlib.patches as mpatches

consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'

access_token = 'your_access_token'
access_token_secret = 'your_token_secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


class StreamListener(tweepy.StreamListener):
    """
    Class used for retrieving tweets in real-time. Modifies exisitng methods from the tweepy library
    to enable data collection for use later on
    """

    def __init__(self, api, max_tweets):
        self.api = api
        self.me = api.me()
        self.tweets = []
        self.max_tweets = max_tweets
        self.counter = 0

    def get_tweets(self):
        return self.tweets

    def on_status(self, status):
        if self.counter == self.max_tweets:
            return False

        if not hasattr(status, 'retweeted_status'):
            self.counter += 1
            self.tweets.append(status)
            return True

    def on_error(self, status_code):
        if status_code == 420:
            return False
        print("Error!")


def produce_visualisations(data_frame, query):
    """
    Helper function for producing graphical visualisations, simply
    calls existing functions
    """
    plot_sentiment(data_frame, query)
    plot_classifications(data_frame)


def print_tweets_categosied(data_frame):
    """
    Prints the collected sample of tweets separated by category
    in a particular order depending on the category
    """

    # print neutral tweets
    # these have the same polarity score of 0, so no need to sort
    count = 0
    print("***\nNEUTRAL TWEETS:")
    for i in range(0, data_frame.shape[0]):
        if data_frame['Classification'][i] == 'Neutral':
            print(str(count + 1) + ') ' + data_frame['User tweets'][i])
            count += 1
    if count == 0:
        print("No neutral tweets!\n***\n")

    # sort and print negative tweets descendingly
    data_frame = data_frame.sort_values(by=['Polarity score'], ascending=False)
    count = 0
    print("\n***\nNEGATIVE TWEETS:")
    for i in range(0, data_frame.shape[0]):
        if data_frame['Classification'][i] == 'Negative':
            print(str(count + 1) + ') ' + data_frame['User tweets'][i])
            print('Polarity:' + str(data_frame['Polarity score'][i]))
            count += 1
    if count == 0:
        print("No neutral tweets!\n***\n")

    # sort and print positive tweets ascendingly
    data_frame = data_frame.sort_values(by=['Polarity score'])
    count = 0
    print("\n***\nPOSITIVE TWEETS:")
    for i in range(0, data_frame.shape[0]):
        if data_frame['Classification'][i] == 'Positive':
            print(str(count + 1) + ') ' + data_frame['User tweets'][i])
            print('Polarity:' + str(data_frame['Polarity score'][i]))
            count += 1
    if count == 0:
        print("No neutral tweets!\n***\n")


def tweet_stream(query=['2020']):
    """
    Retrieves tweets that are generated live based on specified search terms given in an array.
    Twitter API searches for tweets that have an individual term or combination of them
    """
    stream_listener = StreamListener(api, 30)
    stream = tweepy.Stream(api.auth, stream_listener)
    stream.filter(track=query, languages=['en'])

    data_frame = process_tweets(stream_listener.get_tweets(), 'stream')
    report_stats(data_frame)
    produce_visualisations(data_frame, query)
    print_tweets_categosied(data_frame)
    generate_graph(data_frame, query)
    word_data = ' '.join([tweet for tweet in data_frame['User tweets']])
    word_cloud_generate(word_data)


def process_tweets(tweets, action):
    """
    Collates retrieved tweets into a data frame matrix and performs
    data processing
    """
    data_frame = None

    if action == 'stream':
        data = []
        for tweet in tweets:
            if tweet.truncated:
                data.append(tweet.extended_tweet['full_text'])
            else:
                data.append(tweet.text)
        data_frame = pd.DataFrame(data, columns=['User tweets'])
    elif action == 'scan user':
        data_frame = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['User tweets'])

    data_frame['User tweets'] = data_frame['User tweets'].apply(filter_text)
    data_frame['Subjectivity score'] = data_frame['User tweets'].apply(get_subjectivity)
    data_frame['Polarity score'] = data_frame['User tweets'].apply(get_polarity)
    data_frame['Classification'] = data_frame['Polarity score'].apply(get_classification)

    return data_frame


def get_user_tweets(screen_name):
    """
    Retrieves the 100 most recent tweets of a user (excludes retweets)
    """

    tweets = []
    posts = api.user_timeline(id=screen_name, count=200, lang='en', tweet_mode='extended')
    counter = 0

    for tweet in posts:
        if not hasattr(tweet, 'retweeted_status'):
            tweets.append(tweet)
            counter += 1
        if counter == 100:
            break

    data_frame = process_tweets(tweets, 'scan user')
    report_stats(data_frame)
    produce_visualisations(data_frame, screen_name)
    print_tweets_categosied(data_frame)
    word_data = ' '.join([tweet for tweet in data_frame['User tweets']])
    word_cloud_generate(word_data)


def plot_sentiment(data_frame, query):
    """
    Graph the subjectivity and polarity scores of tweets on a scatter plot
    Each metric is measured on one axis
    """

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12, 12))
    for i in range(0, data_frame.shape[0]):
        plt.scatter(data_frame['Polarity score'][i], data_frame['Subjectivity score'][i], color='Blue')

    terms = ''
    if not isinstance(query, list):
        terms = query
    elif len(query) == 1:
        terms = query[0]
    elif len(query) == 2:
        terms = query[0] + ' and/or ' + query[1]
    else:
        for i in range(0, len(query) - 1):
            term = query[i] + ', '
            terms += term
        terms += ' and/or {}'.format(query[-1])

    if not isinstance(query, list):
        plt.title("Sentiment analysis of recent tweets made by {}".format(terms))
    else:
        plt.title("Sentiment analysis of recent tweets about {}".format(terms))
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.show()


def report_stats(data_frame):
    """
    Provides basic stats after a tweet sentiment analysis is performed
    including average metric scores, counts, and proporitons
    """
    print('\n***STAT OVERVIEW***\n')
    avg_polarity = round(data_frame['Polarity score'].mean(), 2)
    avg_subjectivity = round(data_frame['Subjectivity score'].mean(), 2)

    positive_tweets = data_frame[data_frame.Classification == 'Positive']['User tweets'].shape[0]
    negative_tweets = data_frame[data_frame.Classification == 'Negative']['User tweets'].shape[0]
    neutral_tweets = data_frame[data_frame.Classification == 'Neutral']['User tweets'].shape[0]
    total_tweets = data_frame.shape[0]

    print('Average polarity: ' + str(avg_polarity))
    print('Average subjectivity: ' + str(avg_subjectivity))

    print('\nClassification proportions')
    print('Neutral: {}%'.format(round(neutral_tweets / total_tweets * 100), 2))
    print('Positive: {}%'.format(round(positive_tweets / total_tweets * 100), 2))
    print('Negative: {}%'.format(round(negative_tweets / total_tweets * 100), 2))

    print('\nClassification count')
    print(data_frame['Classification'].value_counts())


def plot_classifications(data_frame):
    """
    Plots in a bar graph the polarity classification counts
    """
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    data_frame['Classification'].value_counts().plot(kind='bar')
    plt.show()


def word_cloud_generate(word_data):
    """
    Generates a word cloud based on the sample of tweets in the outline of the Twitter logo
    """
    cwd = os.getcwd()
    img_mask = np.array(Image.open(os.path.join(cwd, "twitter_logo.png")))

    word_cloud = WordCloud(width=700, height=500, max_font_size=150, background_color='white', mask=img_mask).generate(
        word_data)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def generate_graph(data_frame, query):
    """
    Generates a connected scatter plot to map changes in the polarity score of a particular term over a given number of tweets
    Method written to work for one term but can easily be modified to work for multiple terms
    """

    plt.style.use('ggplot')
    labels = query
    colors = ['b']

    fig = plt.figure(figsize=(2, 1.5))
    patches = [mpatches.Patch(color=color, label=label) for label, color in zip(labels, colors)]
    fig.legend(patches, labels, loc='upper left')

    plt.plot(data_frame['Polarity score'], label='Trump', color=colors[0], linestyle='-', marker='o', markersize=8)
    plt.title('Twitter sentiment analysis about {}'.format(labels[0]))
    plt.xlabel('Number of tweets', fontsize=12)
    plt.ylabel('Polarity value (Scored from -1 to 1)', fontsize=12)
    plt.tight_layout()
    plt.show()


def get_polarity(text):
    '''
    Returns the polarity score of a tweet
    '''
    return TextBlob(text).sentiment[0]


def get_subjectivity(text):
    """
    Returns the subjectivity score of a tweet
    """
    return TextBlob(text).sentiment[1]


def get_classification(polarity_score):
    """
    Qualitatively classifies the polarity of a tweet based on its quantitative score
    as determined by Textblob
    """
    if polarity_score == 0:
        return 'Neutral'
    elif polarity_score > 0:
        return 'Positive'
    return 'Negative'


def filter_text(text):
    """
    Removes mentions, hashtags, and hyperlinks from text
    as these interfere with NLP and Textblob's ability to analyse text
    """
    text = re.sub('@[A-Za-z0–9]+', '', text)
    text = re.sub('#', '', text)
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub('&amp', '&', text)
    return text