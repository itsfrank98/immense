import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from os.path import exists
from keras.models import load_model
from text_preprocessing import TextPreprocessing
from word_embedding import WordEmb
from ae import AE
from graph_embeddings.node2vec import Node2VecEmbedder
from decision_tree import *
from utils import prepare_for_decision_tree, create_or_load_tweets_list


seed = 123
np.random.seed(seed)


if __name__ == "__main__":
    df_tweets = pd.read_csv('tweet_labeled_full.csv', sep=',')
    '''X_train, X_test, Y_train, Y_test = train_test_split(df_tweets['id'].to_list(), df_tweets['label'].to_list(),
                                                        test_size=0.2, train_size=0.8)'''
    # train W2V on Twitter dataset
    tok = TextPreprocessing()
    tweet = tok.token_list(df_tweets['text_cleaned'].tolist())
    w2v_model = WordEmb(tweet)

    # split tweet in safe and dangerous
    dang_tweets = df_tweets.loc[df_tweets['label'] == 1]['text_cleaned']
    safe_tweets = df_tweets.loc[df_tweets['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    if not exists('model/w2v_text.h5'):
        w2v_model.train_w2v()
    #print(list_dang_tweets)
    # convert text to vector
    list_dang_tweets = create_or_load_tweets_list(path='textual_data/list_dang_tweets.pickle', w2v_model=w2v_model, tokenized_list=list_dang_tweets)
    list_safe_tweets = create_or_load_tweets_list(path='textual_data/list_safe_tweets.pickle', w2v_model=w2v_model, tokenized_list=list_safe_tweets)

    #################TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER####################
    dang_ae = AE(input_len=512, X_train=list_dang_tweets, label='dang').load_autoencoder()
    safe_ae = AE(input_len=512, X_train=list_safe_tweets, label='safe').load_autoencoder()

    ################# TRAIN OR LOAD DECISION TREES ####################
    rel_tree_path = "model/dtree_rel.h5"
    clos_tree_path = "model/dtree_clos.h5"
    path_to_edges_rel = "graph_embeddings/stuff/network.edg"
    path_to_edges_clos = "graph_embeddings/stuff/closeness_network.edg"
    rel_n2v_path = "model/n2v_rel.h5"
    clos_n2v_path = "model/n2v_clos.h5"

    n2v_rel = Node2VecEmbedder(path_to_edges=path_to_edges_rel, weighted=False, directed=True, n_of_walks=10,
                               walk_length=10, embedding_size=128, p=1, q=4, epochs=100, model_path=rel_n2v_path,
                               name="relationships").load_model()
    if not exists(rel_tree_path):   # IF THE DECISION TREE HAS NOT BEEN LEARNED, LOAD/TRAIN THE N2V MODEL
        train_set_ids_rel = [i for i in df_tweets['id'] if str(i) in n2v_rel.wv.key_to_index]
        difference_rel = set(df_tweets['id']).symmetric_difference(train_set_ids_rel)   # List of users for which we don't have information about their relationships with other users
        train_decision_tree(train_set_ids=train_set_ids_rel, save_path=rel_tree_path, n2v_model=n2v_rel,
                            train_set_labels=df_tweets[df_tweets['id'].isin(train_set_ids_rel)]['label'], name="relationships")
        if difference_rel:
            np.save("model/difference_rel", difference_rel)

    n2v_clos = Node2VecEmbedder(path_to_edges=path_to_edges_clos, weighted=True, directed=False, n_of_walks=10,
                                walk_length=10, embedding_size=128, p=1, q=4, epochs=100, model_path=clos_n2v_path,
                                name="closeness").load_model()
    if not exists(clos_tree_path):
        train_set_ids_clos = [i for i in df_tweets['id'] if str(i) in n2v_clos.wv.key_to_index]
        difference_clos = set(df_tweets['id']).symmetric_difference(train_set_ids_clos)   # List of users for which we don't have information about their closeness to other users
        train_decision_tree(train_set_ids=train_set_ids_clos, save_path=clos_tree_path, n2v_model=n2v_clos,
                            train_set_labels=df_tweets[df_tweets['id'].isin(train_set_ids_clos)]['label'], name="closeness")
        if difference_clos:
            np.save("model/difference_clos", difference_clos)

    tree_rel = load_decision_tree(rel_tree_path)
    tree_clos = load_decision_tree(clos_tree_path)
    difference_rel = difference_clos = []
    if exists("model/difference_rel.npy"):
        difference_rel = np.load("model/difference_rel.npy", allow_pickle=True)
    if exists("model/difference_clos.npy"):
        difference_clos = np.load("model/difference_rel.npy", allow_pickle=True)

    ################# NOW THAT WE HAVE THE MODELS WE CAN OBTAIN THE TRAINING SET FOR THE MLP #################
    list_tweets_embs = w2v_model.text_to_vec(tweet)
    prediction_dang = dang_ae.predict(list_tweets_embs)
    prediction_safe = safe_ae.predict(list_tweets_embs)

    tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(list_tweets_embs, dtype=tf.float32)).numpy()
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, tweets_sigmoid).numpy()
    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, tweets_sigmoid).numpy().tolist()

    for index, row in df_tweets.iterrows():
        id = row['id']
        if id not in difference_rel:
            pr, confidence = test_decision_tree(test_set_ids=[id], test_set_labels=row['label'], cls=tree_rel, n2v_model=n2v_rel)
        else:
            pr, confidence = row['label'], 1.0


