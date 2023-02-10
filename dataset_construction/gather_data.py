import requests
import tweepy
import datetime
from keywords_l import *
import json
import numpy as np

def retrieve_tweets_requests():
    url = "https://counter.ricerca.sesar.di.unimi.it/"
    query = {'start': "20230205", 'end': "20230205"}
    url += "twitter/search/"
    for w in ['Alienation']:
        print(w)
        url += w+"/20230205/20230205"
        response = requests.get(url, params=query)
        print(response.text)

client = tweepy.Client("AAAAAAAAAAAAAAAAAAAAADQFdgEAAAAAL8i70iwJKu%2B%2Fm9rtk70nvc%2BTH10%3D64PhcMyDUJksm9hhENJc7Xa4DbEAkJQzjcDz9MX2PN20OOf0h3")

tweet_fields = ['text', 'created_at', 'author_id', 'geo', 'conversation_id']
place_fields = ['geo', 'id', 'country']
start = datetime.datetime(year=2022, month=2, day=6, hour=0, minute=0, second=0)
end = datetime.datetime(year=2023, month=2, day=6, hour=23, minute=59, second=59)

tweets = []
for w in ['Incel']:
    '''tweets = tweepy.Paginator(client.search_all_tweets, query=("(Non so cosa scrivere questa è una prova) lang:en".format(w)), max_results=10,
                                tweet_fields=tweet_fields, place_fields=place_fields, start_time=start, end_time=end).flatten(limit=10)'''
    pages = tweepy.Paginator(client.search_all_tweets, query="Non so cosa scrivere questa è una prova lang:it", max_results=500, tweet_fields=tweet_fields,
                                place_fields=place_fields, expansions=["geo.place_id"], start_time=start, end_time=end)

    if pages:
        for page in pages:
            for result in page.data:
                tweet = {'id': result['id'],
                         'text': result['text'],
                         'posted': datetime.datetime.timestamp(result['created_at']),
                         'author_id': result['author_id'],
                         'conversation_id': result['conversation_id']}
                if result['geo']:
                    tweet['geo'] = result['geo']['place_id']
                tweets.append(tweet)
            if page.includes:
                print(page.includes['places'])
                for p in page.includes['places']:
                    id = p.id
                    for t in tweets:
                        if t['geo'] and t['geo'] == id:
                            #print(p.geo['bbox'])
                            t['lat'] = np.mean([p.geo['bbox'][1], p.geo['bbox'][3]])
                            t['lon'] = np.mean([p.geo['bbox'][0], p.geo['bbox'][2]])
                            t['country'] = p.country

        #final = json.dumps(tweets, indent=2)
        with open("tweets.json", 'w') as f:
            json.dump(tweets, f, indent=2)
