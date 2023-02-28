import requests
from dataset_construction.keywords_l import list_keyword
from tqdm import tqdm
import json
import os

headers= {
    'bearer_token': "AAAAAAAAAAAAAAAAAAAAADQFdgEAAAAAL8i70iwJKu%2B%2Fm9rtk70nvc%2BTH10%3D64PhcMyDUJksm9hhENJc7Xa4DbEAkJQzjcDz9MX2PN20OOf0h3",
    'access_token': "1522981373161713666-tVg3YuH27pkGsU8cwj0VlOn3lX3T39",
    'access_token_secret': "HxTISDBEyyJN6HpUQ438uCZDEOUzMywAoQe8BpcC9gQN9",
    'api_key': "aVpnQeriuh70jP0ytidnUAhW1",
    'api_key_sec': "ZZH3If7c2ezMLNbr29fQchtI2eydr3heb2e5HRDBKkubRzX1jL",
    'account_type': "academic",
    'skip': 'True'}


def process_request(url, d, i, dict_path):
    """
    :param url:
    :param d:
    :param i: Item id
    :param dict_path: Path where the dictionary will be serialized
    :return:
    """
    req = requests.get(url, headers=headers)
    if req.status_code == 200:
        keys = list(req.json().keys())
        first_key = keys[0]
        if len(req.json()[first_key]) > 0:
            d[i] = req.json()['zipfile']
            with open("{}".format(dict_path), 'w') as f:
                json.dump(d, f, indent=2)
        else:
            print(req.json())
            print("The query for the item '{}' returned 0 results".format(i))
    else:
        print("The query for the item '{}' returned status code {}".format(i, req.status_code))


def search_tweets(links_filename):
    d = {}
    for w in tqdm(list_keyword):
        url = "http://172.20.28.128:8000/twitter/search/"
        if w in ['Alienation', 'Alpha', 'Abroad', 'Absence of social media activity']:
            continue
        print(w)
        if len(w.split()) > 2:
            url += '{}'.format(w)    # If the keyword contains more than 2 words, search the tweets with those words, in whatever order
        else:
            url += '"{}"'.format(w)  # Otherwise search for the tweets containing the exact phrase
        url += " lang:en"
        process_request(url, d, w, links_filename)


def search_followers(dir):
    users_ids = []
    d = {}
    for f in os.listdir(dir):
        with open(os.path.join(dir, f), 'rb') as fp:
            tw = json.load(fp)
        user_id = tw['author_id']
        if user_id not in users_ids:
            users_ids.append(user_id)
            url = "http://172.20.28.128:8000/twitter/followers/{}".format(user_id)
            process_request(url, d, user_id, "followers_links.json")


if __name__ == "__main__":
    search_tweets("tweets_links.json")
