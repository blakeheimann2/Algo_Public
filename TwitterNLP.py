import pandas as pd
import tweepy as tw
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

nltk.download('punkt')

consumer_key= '##################'
consumer_secret= '###################'
access_token= '###########################'
access_token_secret= '##################################'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def get_tweets(text, since_date, count):
    search_term = str(text+"-filter:retweets")
    tweets = tw.Cursor(api.search, q=search_term, lang="en", since=since_date).items(count)
    all_tweets = [tweet.text for tweet in tweets]
    #all_tweets = pd.DataFrame(all_tweets)
    return all_tweets

def remove_url(text):
   return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())


def pre_process_tweet(tweet):
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)
    return [word for word in tweet if not word in stopwords]

spx_tweets = get_tweets('SPX','2019-01-01',1000)
#preprocessing
'''
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
for tweet in list(spx_tweets):
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)


all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
tweets_nsw = [[word for word in tweet_words if not word in stop_words] for tweet_words in words_in_tweet]

'''
#preprocessing
tweets = pd.Dataframe(spx_tweets)
import re
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

X = tweets.astype(str)
X.columns = ['tweet']

#remove stopwords
stop = stopwords.words('english') #type stop to see your stopwords in a list
X.tweet = X.tweet.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
X = X.tweet
X.tail()

documents = []

for sen in range(0, len(X)):
    # remove twitter handle
    document = re.sub('@[^\s]+', '', str(X[sen]))  # X[sen] first because loop

    # remove any urls
    document = re.sub(r"http\S+", "", document)

    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization - converting words with ing/es/ed to their main word
    document = document.split()
    document = [lmtzr.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

tweet_df = pd.DataFrame(X)
tweet_df['clean_tweet'] = documents


from afinn import Afinn
af = Afinn()

sentiment_scores = [af.score(tweet) for tweet in documents]
sentiment_scores



import matplotlib.pyplot as plt
plt.hist(tweet_df.sentiment)
plt.show()sentiment_category = ['positive' if score > 0
                          else 'negative' if score < 0
                              else 'neutral'
                                  for score in sentiment_scores]

tweet_df['sentiment'] = sentiment_scores
tweet_df['sentiment_cat'] = sentiment_category
tweet_df.head()

tweet_df['sentiment_cat'].value_counts().plot(kind='bar')

overall_sentiment = tweet_df.sentiment.sum()
overall_sentiment

#split the messages up by word for topic modelling
docs_split = []
for sen in range(0, len(documents)):
    document = documents[sen].split()
    docs_split.append(document)

from gensim import corpora
dictionary = corpora.Dictionary(docs_split)
corpus = [dictionary.doc2bow(text) for text in docs_split]


import gensim
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
#ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=8)
for topic in topics:
    print(topic)


model_topics = pd.DataFrame(topics)

topic_pred = []
for i in range(0,len(docs_split)):
    doc = dictionary.doc2bow(docs_split[i])
    pred = ldamodel.get_document_topics(doc)
    topic_pred.append(pred)

topic_df = pd.DataFrame(documents)
topic_df['topic'] = topic_pred
topic_df.head(15)
results = pd.merge(tweet_df, topic_df, how='inner', left_on = tweet_df.index, right_on=topic_df.index)
results