from stellargraph.data import BiasedRandomWalk
from graph_embeddings.graph import Graph
from gensim.models import Word2Vec


class Node2VecEmbedder(Graph):
    def __init__(self, path_to_edges, adj_matrix, number_of_walks, walk_length, num_samples, embedding_size, p, q,
                 feature_attribute_name=None):
        # p Defines probability, 1/p, of returning to source node
        # q Defines probability, 1/q, for moving to a node away from the source node
        super().__init__(path_to_edges, adj_matrix, feature_attribute_name)
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.num_samples = num_samples
        self.embedding_size = embedding_size
        self.p = p
        self.q = q

    def learn_n2v_embeddings(self, save=None):
        """
        :param save: string containing the path to which the learned model will be saved. If you don't want to save the
        model, set this parameter to none.
        """
        walker = BiasedRandomWalk(self.sgraph)
        walks = walker.run(n=self.number_of_walks, nodes=list(self.sgraph.nodes()), length=self.walk_length, p=self.p, q=self.q)
        model = Word2Vec(sentences=walks, vector_size=self.embedding_size, window=4, min_count=0, sg=1)
        if save:
            model.save(save)
