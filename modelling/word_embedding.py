from gensim.models import Word2Vec
import numpy as np
from os.path import exists, join
seed = 123
np.random.seed(seed)


class WordEmb:
    def __init__(self, token_word, embedding_size, window, epochs, model_dir):
        self._token_word = token_word
        self._word_vec_dict = {}
        self._model_path = join(model_dir, 'w2v_text.h5')
        self.embedding_size = embedding_size
        self.window = window
        self.epochs = epochs

    def load_dict(self):
        if exists(self._model_path):
            w2v_model = self.load_model()
            vocab = w2v_model.wv.index_to_key
            for word in vocab:
                self._word_vec_dict[word] = w2v_model.wv.get_vector(word)
        else:
            print('Please train W2V model.')

    def train_w2v(self):
        if not exists(self._model_path):
            w2v_model = Word2Vec(vector_size=self.embedding_size, seed=seed, window=self.window, min_count=0, sg=1, workers=1)
            w2v_model.build_vocab(self._token_word, min_count=1)
            total_examples = w2v_model.corpus_count
            w2v_model.train(self._token_word, total_examples=total_examples, epochs=self.epochs)
            w2v_model.save(self._model_path)

    def load_model(self):
        return Word2Vec.load(self._model_path)

    def text_to_vec(self, tweets):
        self.load_dict()
        list_tot = []
        for tw in tweets:
            list_temp = []
            for t in tw:
                embed_vector = self._word_vec_dict.get(t)
                if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                    list_temp.append(embed_vector)
                else:
                    list_temp.append(np.zeros(shape=(self.embedding_size)))
            list_temp = np.array(list_temp)
            #print(list_temp.shape)
            list_temp = np.sum(list_temp, axis=0)
            list_tot.append(list_temp)
        list_tot = np.asarray(list_tot)
        return list_tot
