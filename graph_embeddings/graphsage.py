import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph import globalvar
from stellargraph import datasets
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from stellargraph.mapper import GraphSAGENodeGenerator
from sklearn.metrics import accuracy_score
from graph import Graph
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy


class GraphSAGEEmbedder(Graph):
    def __init__(self, path_to_edges, adj_matrix, embedding_size, number_of_walks, walk_length, samples_per_hop: list, feature_attribute_name=None):
        super().__init__(path_to_edges, adj_matrix, feature_attribute_name)
        self.embedding_size = embedding_size
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.samples_per_hop = samples_per_hop

    def fit_model(self, batch_size, layer_sizes, epochs, dropout=0.0, lr=1e-3, start_nodes=None):
        """
        Sample couple of positive and negative edges and use them to train a link classification model
        :param batch_size:
        :param layer_sizes:
        :param dropout:
        :param epochs:
        :param lr:
        :return:
        """
        num_samples = [10, 5]
        if not start_nodes:
            start_nodes = self.sgraph.nodes()
        unsupervised_samples = UnsupervisedSampler(self.sgraph, nodes=start_nodes, length=self.walk_length, number_of_walks=self.number_of_walks)
        generator = GraphSAGELinkGenerator(self.sgraph, batch_size, num_samples)
        train_gen = generator.flow(unsupervised_samples)
        graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=dropout, normalize="l2")   # encoder, produces embeddings
        x_inp, x_out = graphsage.in_out_tensors()
        #print(x_inp, x_out)
        prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)     # gets the outputs of the encoder and performs link prediction to state if two nodes can be linked
        model = keras.Model(x_inp, prediction)
        model.compile(optimizer=Adam(lr), loss=binary_crossentropy, metrics=binary_accuracy)
        model.fit(train_gen, epochs=epochs, verbose=1, workers=4, shuffle=True)

        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
        node_ids = self.sgraph.nodes()
        node_gen = GraphSAGENodeGenerator(self.sgraph, batch_size, num_samples).flow(node_ids)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
        print(node_embeddings)


if __name__ == "__main__":
    g = GraphSAGEEmbedder("stuff/network.dat", "stuff/adj_net.csv", feature_attribute_name="attribute", embedding_size=10, number_of_walks=1, walk_length=5,
                          samples_per_hop=[2, 3])
    g.create_unweighted_graph()
    g.instanciate_graph()
    #print(g.sgraph.nodes()[1])
    g.create_unweighted_graph()
    g.add_features("const", 10)
    g.instanciate_graph()
    g.fit_model(512, layer_sizes=[128, 128], epochs=10)




