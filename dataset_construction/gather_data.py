import pickle
import time
import tweepy
import datetime
import requests
from keywords_l import *
import json
import numpy as np
from tqdm import tqdm
import os


def retrieve_tweets_requests():
    url = "https://counter.ricerca.sesar.di.unimi.it/"
    query = {'start': "20230205", 'end': "20230205"}
    url += "twitter/search/"
    for w in ['Alienation']:
        print(w)
        url += w+"/20230205/20230205"
        response = requests.get(url, params=query)
        print(response.text)


def retrieve_tweets_dang_words(client, tweet_fields, place_fields):
    """Look for tweets containing the words in the list of dangerous expressions"""
    start = datetime.datetime(year=2022, month=2, day=6, hour=0, minute=0, second=0)
    end = datetime.datetime(year=2023, month=2, day=6, hour=23, minute=59, second=59)

    for w in tqdm(list_keyword):
        tweets = []
        if len(w.split()) > 3:
            query = '{}'.format(w)   # If the keyword contains more than 3 words, search the tweets with those exact expressions
        else:
            query = '"{}"'.format(w)  # Otherwise search for the tweets containing the words in whatever order

        connection = False
        while not connection:
            try:
                pages = tweepy.Paginator(client.search_all_tweets, query=query+" lang:en", max_results=500,
                                         tweet_fields=tweet_fields, place_fields=place_fields,
                                         expansions=["geo.place_id"], start_time=start, end_time=end, limit=4)
                for page in pages:
                    if page.data:
                        for result in page.data:
                            tweet = {'id': result['id'],
                                     'text': result['text'],
                                     'posted': datetime.datetime.timestamp(result['created_at']),
                                     'author_id': result['author_id'],
                                     'in_reply_to_user_id': result['in_reply_to_user_id'],
                                     'entities': result['entities']}
                            if result['geo']:
                                tweet['geo'] = result['geo']['place_id']
                            tweets.append(tweet)
                        if page.includes:       # If there is any additional information about the position
                            for p in page.includes['places']:
                                id = p.id
                                for t in tweets:       # Scan through the list of retrieved tweets
                                    if 'geo' in t.keys():   # If the current tweet has the geo attribute, add info
                                        if t['geo'] == id:
                                            #print(p.geo['bbox'])
                                            t['lat'] = np.mean([p.geo['bbox'][1], p.geo['bbox'][3]])
                                            t['lon'] = np.mean([p.geo['bbox'][0], p.geo['bbox'][2]])
                                            t['country'] = p.country

                        # Read the file containing the tweets retrieved so far
                        l = []
                        if os.path.exists("tweets_tweepy_entities.json"):
                            with open("tweets_tweepy_entities.json", 'r') as f:
                                l = json.load(f)
                                f.close()
                        l.append(tweets)    # Add the tweets retrieved in the current step
                        # Save it
                        with open("tweets_tweepy_entities.json", 'w') as f:
                            json.dump(l, f, indent=2)

            except requests.exceptions.ConnectionError as e:
                print("No connection")      # If there is no connection, wait 60 seconds and retry
                time.sleep(60)
            except tweepy.errors.Unauthorized as e:
                print("unauthorized")
                continue


def retrieve_users(tweets_ids):
    private_key = "AAAAAAAAAAAAAAAAAAAAAJjJlgEAAAAAz%2B2JNuxAQbGkGIikXj2MUzmCPK8%3DYsbmk5BFYPWUMlLteee4BdnQKl9C2GUbxVFxRCfXyMJ56CzbR8"
    acad_key = "AAAAAAAAAAAAAAAAAAAAADQFdgEAAAAAL8i70iwJKu%2B%2Fm9rtk70nvc%2BTH10%3D64PhcMyDUJksm9hhENJc7Xa4DbEAkJQzjcDz9MX2PN20OOf0h3"
    user_fields = ['username', 'created_at', 'location', 'verified']
    #users = {}
    with open("users.json", 'rb') as f:
        users = json.load(f)
    count_acad = 0
    count_private = 0
    key = acad_key
    client = tweepy.Client(key)
    '''l = len(tweets_ids)
    c = 0'''
    for tw_id in tqdm(sorted(tweets_ids)):
        '''c += 1
        if c%15 == 0:
            print("{}/{}".format(c, l))'''
        tweets_ids.remove(tw_id)
        if key == acad_key:
            count_acad += 1
            if count_acad == 76:
                key = private_key
                client = tweepy.Client(key)
        if key == private_key:
            count_private += 1
            if count_private == 76:     # Both the keys have reached the 15 minutes limit. Put the program on sleep for 15 minutes
                count_private = 0
                time.sleep(15*60)
                key = acad_key
                count_acad = 1
                client = tweepy.Client(key)

        connection = False
        while not connection:
            try:
                retweeters = client.get_retweeters(tw_id, user_fields=user_fields, max_results=15)
                likers = client.get_liking_users(tw_id, user_fields=user_fields, max_results=15)
                connection = True
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
                            if 'retweeted' not in users[user_id].keys():
                                users[user_id]['retweeted'] = [tw_id]
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
                with open("users.json".format(tw_id), 'w') as f:
                    json.dump(users, f, indent=2)
                with open("left_ids.pkl", 'wb') as f:
                    pickle.dump(tweets_ids, f)
            except requests.exceptions.ConnectionError as e:
                print("No connection")      # If there is no connection, wait 60 seconds and retry
                time.sleep(60)
            except tweepy.errors.Unauthorized as e:
                print("unauthorized for tweet {}".format(tw_id))
                continue


if __name__ == "__main__":
    '''with open("left_ids.pkl", 'rb') as f:
        tweets_ids = pickle.load(f)
    retrieve_users(tweets_ids)'''
    key = "AAAAAAAAAAAAAAAAAAAAADQFdgEAAAAAL8i70iwJKu%2B%2Fm9rtk70nvc%2BTH10%3D64PhcMyDUJksm9hhENJc7Xa4DbEAkJQzjcDz9MX2PN20OOf0h3"
    client = tweepy.Client(key)

    tweet_fields = ['text', 'created_at', 'author_id', 'geo', 'in_reply_to_user_id', 'entities']
    place_fields = ['geo', 'id', 'country']
    retrieve_tweets_dang_words(client, tweet_fields=tweet_fields, place_fields=place_fields)

