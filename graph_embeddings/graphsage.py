from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from stellargraph.mapper import GraphSAGENodeGenerator
from sklearn.metrics import accuracy_score
from graph import Graph
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.models import save_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


class GraphSAGEEmbedder(Graph):
    def __init__(self, path_to_edges, adj_matrix, number_of_walks, walk_length, num_samples, layer_sizes, samples_per_hop: list, feature_attribute_name=None):
        super().__init__(path_to_edges, adj_matrix, feature_attribute_name)
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.samples_per_hop = samples_per_hop
        self.num_samples = num_samples
        self.layer_sizes = layer_sizes

    def create_graphsage_model(self, batch_size, dropout=0.0, start_nodes=None):
        """
        Sample couple of positive and negative edges and use them to train a link classification model
        :param batch_size:
        :param dropout:
        :param epochs:
        :param lr:
        :return:
        """
        if not start_nodes:
            start_nodes = self.sgraph.nodes()
        unsupervised_samples = UnsupervisedSampler(self.sgraph, nodes=start_nodes, length=self.walk_length, number_of_walks=self.number_of_walks)
        # Generates training data for the encoder model
        generator = GraphSAGELinkGenerator(self.sgraph, batch_size, self.num_samples)
        train_gen = generator.flow(unsupervised_samples)
        graphsage = GraphSAGE(layer_sizes=self.layer_sizes, generator=generator, bias=True, dropout=dropout, normalize="l2")     # encoder, produces embeddings
        return graphsage, train_gen

    def learn_embeddings(self, graphsage, generator, epochs, mode, lr=1e-3):
        x_inp, x_out = graphsage.in_out_tensors()
        #print(x_inp, x_out)
        # get the outputs of the encoder and performs link prediction to state if two nodes can be linked
        model_checkpoint_callback = ModelCheckpoint(
            filepath='models/{}_{}/{epoch:02d}.hdf5'.format(mode, self.layer_sizes[-1]),
            save_weights_only=False,
            monitor='loss',
            mode='min',
            verbose=1,
            save_best_only=True)
        prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)
        model = keras.Model(x_inp, prediction)
        model.compile(optimizer=Adam(lr), loss=binary_crossentropy, metrics=binary_accuracy)
        model.fit(generator, epochs=epochs, verbose=1, workers=4, shuffle=True, callbacks=[model_checkpoint_callback, EarlyStopping(monitor='loss', patience=20)])

        '''x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
        node_ids = self.sgraph.nodes()
        batch_size = 512
        node_gen = GraphSAGENodeGenerator(self.sgraph, batch_size, num_samples).flow(node_ids)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
        print(node_embeddings.shape)'''


if __name__ == "__main__":
    num_samples = [10, 5]
    number_of_walks = 2
    walk_length = 4
    epochs = 1
    layer_sizes = [128, 128]

    g = GraphSAGEEmbedder("stuff/network.dat", "stuff/adj_net.csv", feature_attribute_name="attribute",
                          number_of_walks=number_of_walks, walk_length=walk_length,
                          samples_per_hop=[2, 3], num_samples=num_samples, layer_sizes=layer_sizes)
    g.create_unweighted_graph()
    g.instanciate_graph()
    #print(g.sgraph.nodes()[1])
    g.create_unweighted_graph()
    g.add_features("row", 10)
    g.instanciate_graph()
    graphsage, generator = g.create_graphsage_model(batch_size=512)
    g.learn_embeddings(graphsage=graphsage, generator=generator, epochs=epochs)




