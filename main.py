import pandas as pd
import numpy as np
from ae import AE
import pickle
from sklearn.metrics import classification_report
import tensorflow as tf
from os.path import exists
from keras.models import load_model
from text_preprocessing import TextPreprocessing
from word_embedding import WordEmb

seed = 123
np.random.seed(seed)


if __name__ == "__main__":
    # train W2V on Twitter dataset
    df_tweets = pd.read_csv('tweet_labaled.csv', sep=',')
    tok = TextPreprocessing()
    tweet = tok.token_list(df_tweets['text_cleaned'].tolist())

    exit()
    w2v_model = WordEmb(tweet)

    # split tweet in safe and dangerous
    dang_tweets = df_tweets.loc[df_tweets['label'] == 1]['text_cleaned']
    safe_tweets = df_tweets.loc[df_tweets['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    w2v_exist = exists('model/w2v_text.h5')
    if not w2v_exist:
        w2v_model.train_w2v()

    # convert text to vector
    list_dang_tweets_exist = exists('list_dang_tweets.pickle')
    if list_dang_tweets_exist:
        with open('list_dang_tweets.pickle', 'rb') as handle:
            list_dang_tweets = pickle.load(handle)
    else:
        list_dang_tweets = w2v_model.text_to_vec(list_dang_tweets)
        with open('list_dang_tweets.pickle', 'wb') as handle:
            pickle.dump(list_dang_tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    list_safe_tweets_exist = exists('list_safe_tweets.pickle')
    if list_safe_tweets_exist:
        with open('list_safe_tweets.pickle', 'rb') as handle:
            list_safe_tweets = pickle.load(handle)
    else:
        list_safe_tweets = w2v_model.text_to_vec(list_safe_tweets)
        with open('list_safe_tweets.pickle', 'wb') as handle:
            pickle.dump(list_safe_tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ###############################

    #################train/load autoencoder safe and dang
    model_safe_exist = exists('model/autoencodersafe.h5')
    model_dang_exist = exists('model/autoencoderdang.h5')

    if model_dang_exist:
        model_dang = load_model('model/autoencoderdang.h5')#load existing model
    else:
        AE(input_len=512, X_train=list_dang_tweets, label='dang').train_autoencoder()#train the model
        model_dang = load_model('model/autoencoderdang.h5')

    if model_safe_exist:
        model_safe = load_model('model/autoencodersafe.h5')  # load existing model
    else:
        AE(input_len=512, X_train=list_safe_tweets, label='safe').train_autoencoder()  # train the model
        model_safe = load_model('model/autoencodersafe.h5')
    ##########################

    #test set evaluation
    print('----W2V Counter dataset----')
    df_counter = pd.read_csv('counter_dang.csv', sep=';',  encoding="latin-1")
    list_counter_split_exist = exists('list_counter_split.pickle')
    if list_counter_split_exist:
        with open('list_counter_split.pickle', 'rb') as handle:
            X_test = pickle.load(handle)
    else:
            print('----Preprocessing Counter dataset----')
            list_counter = []
            for s in df_counter['content']:
                f = tok.preprocessing_text(s)
                list_counter.append(f)

            print(list_counter)

            with open('list_counter.pickle', 'wb') as handle:
                pickle.dump(list_counter, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #list_counter = tok.token_list(list_counter)
            print('----End Preprocessing Counter dataset----')
            X_test = w2v_model.text_to_vec(list_counter)
            with open('list_counter_split.pickle', 'wb') as handle:
                pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('----End W2V Counter dataset----')


    print('Start evaluation')
    prediction_safe = model_safe.predict(X_test)
    prediction_dang = model_dang.predict(X_test)

    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, X_test)
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, X_test)

    prediction_loss_dang = prediction_loss_dang.numpy().tolist()
    prediction_loss_safe = prediction_loss_safe.numpy().tolist()

    i = 0
    y_pred = []
    y_true = df_counter['label'].tolist()
    while i<len(prediction_loss_safe):
        if prediction_loss_safe[i] < prediction_loss_dang[i]:
            label = 0
        else:
            label = 1
        y_pred.append(label)
        i = i + 1

    print(classification_report(y_true, y_pred))
    print('end evaluation')





