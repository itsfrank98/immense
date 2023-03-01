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
import re


'''def retrieve_tweets_requests():
    url = "https://counter.ricerca.sesar.di.unimi.it/"
    query = {'start': "20230205", 'end': "20230205"}
    url += "twitter/search/"
    for w in ['Alienation']:
        print(w)
        url += w+"/20230205/20230205"
        response = requests.get(url, params=query)
        print(response.text)'''


def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)


def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def retrieve_tweets_dang_words(client, tweet_fields, place_fields, dir):
    """
    Look for tweets containing the dangerous keywords. For each keyword, a separate file is produced
    :param client: Client which will send the query
    :param tweet_fields: Fields to retrieve from each tweet
    :param place_fields: Fields concerning the place
    :param dir: Name of the directory where the files with the retrieved tweets will be serialized
    """
    start = datetime.datetime(year=2022, month=2, day=6, hour=0, minute=0, second=0)
    end = datetime.datetime(year=2023, month=2, day=6, hour=23, minute=59, second=59)

    for w in tqdm(sorted(list_keyword)):
        d_tweets = {}
        if len(w.split()) > 3:
            query = '{}'.format(w.lower())   # If the keyword contains more than 3 words, search the tweets with those exact expressions
        else:
            query = '"{}"'.format(w.lower())  # Otherwise search for the tweets containing the words in whatever order

        connection = False      # This is done in order to prevent the process from crashing if network goes down
        while not connection:
            try:
                pages = tweepy.Paginator(client.search_all_tweets, query=query+" lang:en", max_results=500,
                                         tweet_fields=tweet_fields, place_fields=place_fields,
                                         expansions=["geo.place_id"], start_time=start, end_time=end, limit=3)
                for page in pages:
                    if page.data:
                        for result in page.data:
                            tweet = {'text': result['text'],
                                     'posted': datetime.datetime.timestamp(result['created_at']),
                                     'author_id': result['author_id'],
                                     'in_reply_to_user_id': result['in_reply_to_user_id'],
                                     'entities': {},
                                     'referenced_tweets': []}
                            if ('entities' in result.keys()) and ('mentions' in result['entities'].keys()):
                                tweet['entities']['mentions'] = result['entities']['mentions']
                            if 'referenced_tweets' in result.keys():
                                references = []
                                for ref in result['referenced_tweets']:
                                    references.append((ref['id'], ref['type']))
                                tweet['referenced_tweets'] = references
                            if result['geo']:
                                tweet['geo'] = result['geo']['place_id']
                            d_tweets[result['id']] = tweet
                        if page.includes:       # If there is any additional information about the position
                            for p in page.includes['places']:
                                id = p.id
                                for k in d_tweets.keys():       # Scan through the list of retrieved tweets
                                    if 'geo' in d_tweets[k].keys():   # If the current tweet has the geo attribute, add info
                                        if d_tweets[k]['geo'] == id:
                                            d_tweets[k]['lat'] = np.mean([p.geo['bbox'][1], p.geo['bbox'][3]])
                                            d_tweets[k]['lon'] = np.mean([p.geo['bbox'][0], p.geo['bbox'][2]])
                                            d_tweets[k]['country'] = p.country
                connection = True
            except requests.exceptions.ConnectionError as e:
                print("No connection")      # If there is no connection, wait 60 seconds and retry
                time.sleep(60)
            except tweepy.errors.Unauthorized as e:
                print("unauthorized")
                continue
        if d_tweets:
            with open(dir+"/{}.json".format(w), 'w') as f:
                json.dump(d_tweets, f, indent=2)
                f.close()


"""def retrieve_users(tweets_ids):
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
                continue"""


def clean_retrieved_files(dir):
    """
    The Twitter API treats retweets as standalone tweets. Hence, the same content may be present more than once, with
    only information about the author changing. This function cleans the tweets set. All the retweets referring to the
    same tweet are grouped into a unique dictionary entrance having as ID the "principal" tweet (If it was not
    downloaded, it is created since all the necessary information can be retrieved from the retweet info). Then, the
    entrance corresponding to the retweet is deleted. Moreover, tweets that were downloaded more than once are removed.
    The function also removes those tweets that don't contain any keyword but were still downloaded for some reason.
    :param dir: Path to the directory containing the json files (one per keyword, named after the keyword they refer to.
    :return:
    """
    tweet_ids = []
    for fname in os.listdir(dir):
        with open(os.path.join(dir, fname), 'rb') as fp:
            d = json.load(fp)
        keyword = fname.split('.')[0]  # The filename is the keyword since we have a separate for each keyword
        dict_keys_list = list(d.keys())
        for k in tqdm(dict_keys_list):
            to_remove = []          # List that'll contain the IDs of the tweets to remove
            if k not in tweet_ids:  # Check if the tweet is already present
                tweet = d[k]
                tweet_ids.append(k)
                if re.search(r"\b{}\b".format(keyword), tweet['text'], flags=re.IGNORECASE):
                    if tweet['referenced_tweets']:  # See if the tweet refers to another tweet
                        for ref in tweet['referenced_tweets']:
                            if ref[1] == "retweeted":  # If the tweet is a retweet
                                id = ref[0]
                                if id in d.keys():  # Retrieve the tweet to which the current tweet refers. First of all, check if it has been downloaded
                                    if 'retweeted_by' not in d[id].keys():
                                        d[id]['retweeted_by'] = [tweet['author_id']]
                                    else:
                                        d[id]['retweeted_by'].append(tweet['author_id'])
                                elif 'entities' in tweet.keys() and 'mentions' in tweet['entities'].keys():
                                    d[id] = {
                                        'text': tweet['text'],
                                        'posted': tweet['posted'],
                                        'author_id': tweet['entities']['mentions'][0]['id'],
                                        'in_reply_to_user_id': None,
                                        'entities': None,
                                        'referenced_tweets': None,
                                        'retweeted_by': [tweet['author_id']]
                                    }
                                to_remove.append(k)
                else:
                    to_remove.append(k)  # If the tweet doesn't contain an exact match with the keyword we delete it
            else:
                to_remove.append(k)  # If the tweet has already been met, then it's a duplicate, so we delete it
            for kr in to_remove:
                d.pop(kr)
        with open(os.path.join(dir, 'adjusted_'+fname), 'w') as fp:
            json.dump(d, fp, indent=2)


def get_followers(user_id, client:tweepy.Client):
    l = []
    '''if last == client_acad:
        client = client_mio
    else:
        client = client_acad'''
    # client = client_acad
    connection = False
    everything_ok = False
    while not connection:
        try:
            followers = client.get_users_followers(id=user_id, max_results=1000)
            if followers.data:
                for follower in followers.data:
                    l.append(follower['id'])

                """except tweepy.errors.TooManyRequests:
                    if client == client_acad:
                        print("switching")
                        client = client_mio
                    else:
                        for i in range(890):
                            if i%10 == 0:
                                print(i)
                            time.sleep(1)"""
                connection = True
        except requests.exceptions.ConnectionError as e:
            print("No connection")      # If there is no connection, wait 60 seconds and retry
            time.sleep(60)
    return l


def find_followers_from_tweets(src_dir, dst_dir, client_mio, client_acad):
    if os.path.exists("done_users.pkl"):
        users = load_from_pickle("done_users.pkl")
    else:
        users = ['1521241894533087232']
    count_mio = 0
    count_acad = 0
    for f in os.listdir(os.path.join(src_dir)):
        with open(os.path.join(src_dir, f), 'rb') as fp:
            tweets = json.load(fp)
        print(f.split(".")[0])
        if os.path.exists(os.path.join(dst_dir, f)):
            with open(os.path.join(dst_dir, f), 'rb') as fp:
                d = json.load(fp)
        else:
            d = {}
        for tw_id in tqdm(tweets.keys()):
            user_id = tweets[tw_id]['author_id']
            if user_id not in users:
                if count_acad == 15:
                    client = client_mio
                    if count_mio == 15:
                        count_mio = 0
                        count_acad = 1
                        client = client_acad
                        time.sleep(60 * 15)
                    else:
                        count_mio += 1
                else:
                    count_acad += 1

                d[user_id] = get_followers(user_id, client=client)
                users.append(user_id)
                save_to_pickle("done_users.pkl", users)
            if 'retweeted_by' in tweets[tw_id].keys():
                for user_id in tweets[tw_id]['retweeted_by']:
                    if user_id not in users:
                        if count_acad == 15:
                            client = client_mio
                            if count_mio == 15:
                                count_mio = 0
                                count_acad = 1
                                client = client_acad
                                time.sleep(60 * 15)
                            else:
                                count_mio += 1
                        else:
                            count_acad += 1
                        d[user_id] = get_followers(user_id, client=client)
                        users.append(user_id)
                        save_to_pickle("done_users.pkl", users)
            with open(os.path.join(dst_dir, f), 'w') as fp:
                json.dump(d, fp, indent=2)


if __name__ == "__main__":
    #### ACADEMIC ########
    bearer_token_acad = "AAAAAAAAAAAAAAAAAAAAADQFdgEAAAAAL8i70iwJKu%2B%2Fm9rtk70nvc%2BTH10%3D64PhcMyDUJksm9hhENJc7Xa4DbEAkJQzjcDz9MX2PN20OOf0h3"
    api_key_acad = "aVpnQeriuh70jP0ytidnUAhW1"
    api_key_secret_acad = "ZZH3If7c2ezMLNbr29fQchtI2eydr3heb2e5HRDBKkubRzX1jL"
    access_token_acad = "1522981373161713666-tVg3YuH27pkGsU8cwj0VlOn3lX3T39"
    access_token_secret_acad = "HxTISDBEyyJN6HpUQ438uCZDEOUzMywAoQe8BpcC9gQN9"
    client_acad = tweepy.Client(bearer_token=bearer_token_acad, consumer_key=api_key_acad,
                                consumer_secret=api_key_secret_acad, access_token=access_token_acad,
                                access_token_secret=access_token_secret_acad, wait_on_rate_limit=False)

    bearer_token_mio = "AAAAAAAAAAAAAAAAAAAAAJjJlgEAAAAA4vuiBa6otiMDv7Wg7LhUFh88ocM%3DTcoMpXfVgMEGBwKH2rzcxlOMR8NGwCaUF3BfJPkszWEWTwizqr"
    api_key_mio = "DA20juwMuN927DZTBeQQIW6AX"
    api_key_secret_mio = "dkQecMcJYUyEHqOQk5gjkHdhvneCGUipXNd8Q4pK7uqorcDYnq"
    access_token_mio = "1621182211331334150-JHAQHA5WuPUUGPx3Un1C7QcMOPjOXo"
    access_token_secret_mio = "LhPzzz82aDQJgpJKYElmj1LosL16fIOIvwuUpGGhhcUGw"
    client_mio = tweepy.Client(bearer_token=bearer_token_mio, consumer_key=api_key_mio,
                               consumer_secret=api_key_secret_mio, access_token=access_token_mio,
                               access_token_secret=access_token_secret_mio, wait_on_rate_limit=False)

    tweet_fields = ['text', 'created_at', 'author_id', 'geo', 'in_reply_to_user_id', 'entities', 'referenced_tweets']
    place_fields = ['geo', 'id', 'country']
    dir = "tweets"
    if not os.path.exists(dir):
        os.makedirs(dir)
    # retrieve_tweets_dang_words(client, tweet_fields=tweet_fields, place_fields=place_fields, dir=dir)
    #clean_retrieved_files(dir)
    find_followers_from_tweets(src_dir="tweets", dst_dir="followers", client_mio=client_mio, client_acad=client_acad)

