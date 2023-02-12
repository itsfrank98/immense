import requests
import tweepy
import datetime
from keywords_l import *
import json
import numpy as np
from tqdm import tqdm
client = tweepy.Client("AAAAAAAAAAAAAAAAAAAAAJjJlgEAAAAAWn%2B8NbDdZCvkq%2FWBynhIrk3V%2FjI%3DjAaq2jpuplkBvb0to9QJFnyve92jLcAlYVLd25fdzIDHis0HWr")

def retrieve_tweets_dang_words():
    """Look for tweets containing the words in the list of dangerous expressions"""
    tweet_fields = ['text', 'created_at', 'author_id', 'geo', 'conversation_id']
    place_fields = ['geo', 'id', 'country']
    start = datetime.datetime(year=2022, month=2, day=6, hour=0, minute=0, second=0)
    end = datetime.datetime(year=2023, month=2, day=6, hour=23, minute=59, second=59)
    
    tweets = []
    for w in list_keyword:
        pages = tweepy.Paginator(client.search_all_tweets, query=w+" lang:en", max_results=500, tweet_fields=tweet_fields,
                                 place_fields=place_fields, expansions=["geo.place_id"], start_time=start, end_time=end, limit=4)
        for page in pages:
            if page.data:
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
                    for p in page.includes['places']:
                        id = p.id
                        for t in tweets:
                            if 'geo' in t.keys():
                                if t['geo'] == id:
                                    #print(p.geo['bbox'])
                                    t['lat'] = np.mean([p.geo['bbox'][1], p.geo['bbox'][3]])
                                    t['lon'] = np.mean([p.geo['bbox'][0], p.geo['bbox'][2]])
                                    t['country'] = p.country
    
    #final = json.dumps(tweets, indent=2)
    with open("potentially_harmful_tweets.json", 'w') as f:
        json.dump(tweets, f, indent=2)


with open("potentially_harmful_tweets.json", 'rb') as f:
    tweets = json.load(f)

tweets_ids = [tweet['id'] for tweet in tweets]
for id in sorted(tweets_ids):
    if id != 1492166639638532098:
        tweets_ids.remove(id)
    else:
        tweets_ids.remove(id)
        break

user_fields = ['username', 'created_at', 'location', 'verified']
users = {}
with open("users.json", 'rb') as f:
    users = json.load(f)

for tw_id in tqdm(sorted(tweets_ids)):
    retweeters = client.get_retweeters(tw_id, user_fields=user_fields, max_results=15)
    likers = client.get_liking_users(tw_id, user_fields=user_fields, max_results=15)
    if retweeters.data:
        for result in retweeters.data:
            user_id = result['id']
            if user_id not in users.keys():
                users[user_id] = {'username': result['username'],
                                  'retweeted': [tw_id],
                                  'created_at': datetime.datetime.timestamp(result['created_at']),
                                  'verified': result['verified']
                                  }
                if result['location']:
                    users[user_id]['location'] = result['location']
            else:
                users[user_id]['retweeted'].append(tw_id)
    if likers.data:
        for like in likers.data:
            user_id = like['id']
            if user_id not in users.keys():
                users[user_id] = {'username': like['username'],
                                  'liked': [tw_id],
                                  'created_at': datetime.datetime.timestamp(like['created_at']),
                                  'verified': like['verified']
                                  }
                if like['location']:
                    users[user_id]['location'] = like['location']
            else:
                if 'liked' in users[user_id].keys():
                    users[user_id]['liked'].append(tw_id)
                else:
                    users[user_id]['liked'] = [tw_id]
    with open("{}.json".format(tw_id), 'w') as f:
        json.dump(users, f, indent=2)
