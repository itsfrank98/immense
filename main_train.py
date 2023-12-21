import argparse

import torch

from exceptions import *
from modelling.sairus import train
import numpy as np
import pandas as pd
from os.path import exists, join
from os import makedirs
from utils import adj_list_from_df
seed = 123
np.random.seed(seed)


def main_train(args=None):
    """textual_content_path = args.textual_content_path
    social_net_path = args.social_net_path
    spatial_net_path = args.spatial_net_path
    rel_adj_mat_path = args.rel_adj_mat_path
    spat_adj_mat_path = args.spat_adj_mat_path
    id2idx_rel_path = args.id2idx_rel_path
    id2idx_spat_path = args.id2idx_spat_path

    rel_technique = args.rel_technique
    spat_technique = args.spat_technique
    word_embedding_size = args.word_embedding_size
    window = args.window
    w2v_epochs = args.w2v_epochs
    spat_node_embedding_size = args.spat_node_embedding_size
    rel_node_embedding_size = args.rel_node_embedding_size
    n2v_epochs_spat = args.n2v_epochs_spat
    n2v_epochs_rel = args.n2v_epochs_rel
    rel_autoenc_epochs = args.rel_ae_epochs
    spat_autoenc_epochs = args.spat_ae_epochs
    models_dir = args.models_dir
    dataset_dir = args.dataset_dir"""
    dataset_dir = join("dataset", "big_dataset")
    graph_dir = join(dataset_dir, "graph")
    models_dir = join(dataset_dir, "models")
    field_id = "id"
    field_text = "text_cleaned"
    consider_rel = True
    consider_spat = True
    competitor = False
    path_rel = join(graph_dir, "social_network_train.edg")
    path_spat = join(graph_dir, "spatial_network_train.edg")

    textual_content_path = join(dataset_dir, "tweet_labeled_full.csv")
    train_path = join(dataset_dir, "train.csv")
    test_path = join(dataset_dir, "test.csv")

    rel_technique = spat_technique = "graphsage"
    rel_adj_mat_path = spat_adj_mat_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")
    social_net_path = join(graph_dir, "social_net.edg")
    spatial_net_path = join(graph_dir, "spatial_net.edg")

    # w2v parameters
    word_embedding_size = 512
    w2v_epochs = 15
    # node emb parameters
    spat_node_embedding_size = rel_node_embedding_size = 256
    epochs_spat = epochs_rel = 10

    if not exists(dataset_dir):
        makedirs(dataset_dir)
    if not exists(models_dir):
        makedirs(models_dir)

    if not exists(train_path) or not exists(test_path):
        df = pd.read_csv(textual_content_path, sep=',')
        ids = list(df[field_id])
        if len(ids) != len(set(ids)):
            print("Found non univocal IDs. I am now resetting them")
            ld = []
            for index, row in df.iterrows():
                d = {field_id: index, field_text: row[field_text], 'label': row['label']}
                ld.append(d)
            df = pd.DataFrame(ld)
        cols = ['index', 'label', field_id, field_text]
        df = df.drop(columns=[col for col in df.columns if col not in cols])
        df = df.sample(frac=1, random_state=1).reset_index()  # Shuffle the dataframe
        df.to_csv(join(dataset_dir, "shuffled_content.csv"))
        idx = round(len(df)*0.8)
        train_df = df[:idx]
        test_df = df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df = train_df.drop(columns=[col for col in train_df.columns if col not in cols])
        test_df = test_df.drop(columns=[col for col in test_df.columns if col not in cols])
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        adj_list_from_df(train_df, social_net_path, join(graph_dir, "social_network_train.edg"))
        adj_list_from_df(test_df, social_net_path, join(graph_dir, "social_network_test.edg"))
        adj_list_from_df(train_df, spatial_net_path, join(graph_dir, "spatial_network_train.edg"), spatial=True)
        adj_list_from_df(test_df, spatial_net_path, join(graph_dir, "spatial_network_test.edg"), spatial=True)
    else:
        train_df = pd.read_csv(train_path)
    nz = len(train_df[train_df.label==1])
    pos_weight = len(train_df) / nz        #1/2284
    neg_weight = len(train_df) / (2*(len(train_df) - nz))        # 1/30356

    if rel_technique.lower() in ["autoencoder", "pca", "none"]:
        if not rel_adj_mat_path:
            raise AdjMatException(lab="rel")
        if not id2idx_rel_path:
            raise Id2IdxException(lab="rel")
    if spat_technique.lower() in ["autoencoder", "pca", "none"]:
        if not spat_adj_mat_path:
            raise AdjMatException(lab="spat")
        if not id2idx_spat_path:
            raise Id2IdxException(lab="spat")

    train(train_df=train_df, model_dir=models_dir, w2v_epochs=w2v_epochs, batch_size=64, field_name_id=field_id,
          field_name_text=field_text, id2idx_path_spat=id2idx_spat_path, path_rel=path_rel, path_spat=path_spat,
          word_emb_size=word_embedding_size, node_emb_technique_spat=spat_technique,
          node_emb_technique_rel=rel_technique, node_emb_size_spat=spat_node_embedding_size,
          node_emb_size_rel=rel_node_embedding_size, weights=torch.tensor([neg_weight, pos_weight]),
          eps_nembs_spat=epochs_spat, eps_nembs_rel=epochs_rel, adj_matrix_path_spat=spat_adj_mat_path,
          adj_matrix_path_rel=rel_adj_mat_path, id2idx_path_rel=id2idx_rel_path, consider_rel=consider_rel,
          consider_spat=consider_spat, competitor=competitor)


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument("--textual_content_path", type=str, required=True, help="Link to the file containing the posts")
    parser.add_argument("--social_net_path", type=str, required=False, default="", help="Link to the file containing the edges in the social network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("--spatial_net_path", type=str, required=False, default="", help="Link to the file containing the edges in the spatial network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("--word_embedding_size", type=int, default=512, required=True, help="Dimension of the word embeddings")
    parser.add_argument("--w2v_epochs", type=int, default=50, required=False, help="For how many epochs to train the w2v model")
    parser.add_argument("--spat_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning spatial node embeddings")
    parser.add_argument("--rel_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning relational node embeddings")
    parser.add_argument("--spat_node_embedding_size", type=int, default=128, required=False, help="Dimension of the spatial node embeddings")
    parser.add_argument("--rel_node_embedding_size", type=int, default=128, required=False, help="Dimension of the relational node embeddings")
    parser.add_argument("--spat_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the spatial adjacency matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("--rel_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the relational adjacency matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("--id2idx_spat_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the spatial adj matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("--id2idx_rel_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the relational adj matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("--n2v_epochs_spat", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn spatial embeddings")
    parser.add_argument("--n2v_epochs_rel", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn relational embeddings")
    parser.add_argument("--rel_ae_epochs", type=int, default=50, required=False, help="Epochs for training the autoencoder that will learn relational embeddings")
    parser.add_argument("--spat_ae_epochs", type=int, default=50, required=False, help="Epochs for training the autoencoder that will learn spatial embeddings")
    parser.add_argument("--models_dir", type=str, default='models', required=False, help="Directory where the models will be saved")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")

    args = parser.parse_args()"""
    args = None
    main_train(args)

