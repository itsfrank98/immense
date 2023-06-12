from gensim.models import Word2Vec, KeyedVectors
from modelling.text_preprocessing import TextPreprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity(matrix, vector):
    # Normalize the matrix rows and the vector
    normat = np.linalg.norm(matrix, axis=1, keepdims=True)
    norvec = np.linalg.norm(vector)
    matrix_norm = matrix / normat
    vector_norm = vector / norvec

    # Calculate the cosine similarity using broadcasting
    similarity = np.dot(matrix_norm, vector_norm)

    return similarity


def text_to_vec(mod_learned:KeyedVectors, posts):
    """
    Obtain a vector from a textual content. This function is needed even if it alrady exists in the WordEmb class, because
    this function works using a pretrained word embedding model"""
    list_tot = []
    e = {w.lower(): w for w in mod_learned.key_to_index.keys() if not w.islower()}
    for tw in posts:
        list_temp = []
        if tw:
            for t in tw:
                try:
                    learned_embedding = mod_learned[t]
                    f_l = True
                except KeyError:
                    try:
                        learned_embedding = mod_learned[e[t]]
                        f_l = True
                    except KeyError:
                        f_l = False
                if f_l:
                    list_temp.append(learned_embedding)
                else:
                    list_temp.append(np.zeros(shape=(300,)))
        else:
            continue
        if list_temp:
            list_temp = np.array(list_temp)
            list_temp = np.sum(list_temp, axis=0)
            list_tot.append(list_temp)
    list_tot = np.asarray(list_tot)
    return list_tot


def plot_values(sorted_values, type, l):
    # Create a list of indices for the elements
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('Index')
    plt.ylabel(type)
    plt.title('{} to Base Array'.format(type))
    labels = [l, "risky"]
    vals_imdb = sorted_values[0][0]
    vals_evil = sorted_values[1][0]
    indices = list(range(len(vals_imdb)+len(vals_evil)))
    ax1.scatter(indices[:len(vals_imdb)], vals_imdb, label=labels[0])
    ax1.scatter(indices[len(vals_imdb):], vals_evil, label=labels[1])
    plt.legend()
    plt.show()


def kfold(dim, splits):
    prog = int(dim/splits)
    folds_idx = np.arange(start=0, stop=dim, step=prog)
    return folds_idx


def main(dataset_negative_path, dataset_to_label_path, model_path, n):
    df_negative = pd.read_csv(dataset_negative_path)
    df_tolabel = pd.read_csv(dataset_to_label_path)        # Dataframe that has to be labeled
    #df_evil = df_evil.sample(frac=1).reset_index()  # Shuffle

    tok = TextPreprocessing()
    posts_content_negative = tok.token_list(df_negative["text"].values.tolist())
    posts_content_tolabel = tok.token_list(df_tolabel["text"].values.tolist())
    model_negative = KeyedVectors.load_word2vec_format(model_path, binary=True)

    folds_idx = kfold(len(df_negative), 5)
    sim_tolabel = np.zeros(len(posts_content_tolabel))
    l_tolabel = np.zeros(shape=(1, len(posts_content_tolabel)))
    for i in range(len(folds_idx)-1):
        # The rows going from cur_idx to next_idx will be used as test fold
        cur_idx = folds_idx[i]
        next_idx = folds_idx[i+1]
        train_tl = posts_content_negative[:cur_idx] + posts_content_negative[next_idx:]     # Token list of the posts that will be used as reference for measuring how risky a post is
        test_tl = posts_content_negative[cur_idx:next_idx]      # Token list of the posts that are known to be negative and that will be compared to those in train_tl

        t2v_negative, a = text_to_vec(posts=train_tl, mod_learned=model_negative)  # Vettore contenente un embedding per ogni tweet risky, ottenuto sommando gli embedding delle parole
        print(a)
        l_negative = np.zeros(shape=(1, len(test_tl)))
        sim_evil = np.zeros(len(test_tl))
        t2v_tolabel, _ = text_to_vec(posts=posts_content_tolabel, mod_learned=model_negative)
        t2v_test_negative, _ = text_to_vec(posts=test_tl, mod_learned=model_negative)
        #print("ciao")
        for j in range(t2v_tolabel.shape[0]):
            sim = cosine_similarity(t2v_negative, t2v_tolabel[j, :])
            sim_tolabel[j] = np.nanmax(sim)
        for j in range(t2v_test_negative.shape[0]):
            sim = cosine_similarity(t2v_negative, t2v_test_negative[j, :])
            sim_evil[j] = np.nanmax(sim)
        l_negative[0, :] = sim_evil
        l_tolabel[0, :] = sim_tolabel

        plot_values([l_tolabel, l_negative], type="cosine", l=n)

    # return l_positive, l_evil"""


if __name__ == "__main__":
    main(dataset_negative_path="evil/no_aff_preproc_nolow.csv", dataset_to_label_path="", model_path="evil/google_w2v.bin",
        n="spam")

