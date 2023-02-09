import requests
import tweepy
import datetime
from keywords_l import *
import json

def retrieve_tweets_requests():
    url = "https://counter.ricerca.sesar.di.unimi.it/"
    query = {'start': "20230205", 'end': "20230205"}
    url += "twitter/search/"
    for w in ['Alienation']:
        print(w)
        url += w+"/20230205/20230205"
        response = requests.get(url, params=query)
        print(response.text)

client = tweepy.Client("AAAAAAAAAAAAAAAAAAAAAJjJlgEAAAAATDw%2Bueed7VLY3hCglqEfO3HU%2BjU%3DkMK0lPWEzNuwY3vclt9iFNHS0cEf6wopV3YwhlCXuqdyxGWbOr")
api = tweepy.API()

tweet_fields = ['text', 'created_at', 'author_id', 'geo']
place_fields = ['lat', 'long', ]
start = datetime.datetime(year=2022, month=12, day=1, hour=0, minute=0, second=0)
end = datetime.datetime(year=2023, month=1, day=31, hour=23, minute=59, second=59)

for w in ['Incel']:
    tweets = tweepy.Paginator(client.search_recent_tweets, query=("({}) lang:en".format(w)), max_results=10,
                                tweet_fields=tweet_fields, start_time=start, end_time=end).flatten(limit=10)
    for t in tweets:
        tweet = {'user_id': tweet.id,
                 'text': tweet.text,
                 'posted': tweet.created_at,
                 'author_id': tweet.author_id,
                 'geo': tweet.geo}
