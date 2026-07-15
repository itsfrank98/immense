import pandas as pd
import yaml
import os
from utils import read_edg_file, write_edg_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def find_isolated_nodes(ids, network_edges):
    network_ids = set()
    for el in tqdm(network_edges):
        network_ids.add(int(el[0]))
        network_ids.add(int(el[1]))
    isolated = set(ids).difference(network_ids)
    return list(isolated)

def split(content_path, rel_edges_fname, id_field):
    df = pd.read_csv(content_path, sep="\t")
    acc_ids = df[id_field].unique().tolist()
    network = read_edg_file(rel_edges_fname, type_pairs="tuple")
    print("Finding isolated nodes...")
    isolated = find_isolated_nodes(acc_ids, network)
    if isolated:
        df_non_isolated = df[~df[id_field].isin(isolated)].sample(frac=1).reset_index(drop=True)
        df_isolated = df[df[id_field].isin(isolated)].sample(frac=1).reset_index(drop=True)

        train_df, test_df = train_test_split(df_non_isolated, test_size=.3, random_state=123)
        train_isolated, test_isolated = train_test_split(df_isolated, test_size=.3, random_state=123)
        train_df = pd.concat([train_df, train_isolated])
        test_df = pd.concat([test_df, test_isolated])
    else:
        df = df.sample(frac=1).reset_index(drop=True)
        train_df, test_df = train_test_split(df, train_size=.7, random_state=123)

    train_network = []
    test_network = []
    train_ids = set(train_df[id_field].tolist())
    test_ids = set(test_df[id_field].tolist())

    for ed in tqdm(network):
        if int(ed[0]) in train_ids and int(ed[1]) in train_ids:
            train_network.append(ed)
        elif int(ed[0]) in test_ids and int(ed[1]) in test_ids:
            test_network.append(ed)

    print(len(network), len(train_network), len(test_network))
    return train_df, test_df, train_network, test_network


if __name__ == "__main__":
    with open("dataset_preprocessing.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)

    content_path = params["content_path"]
    rel_edges_fname = params["rel_edges_fname"]
    dataset_dir = params["output_dataset_dir"]
    graph_dir = params["output_graph_dir"]
    id_field = params["field_name_id"]
    text_field = params["field_name_text"]
    setting = params["setting"]


    train_df, test_df, train_network, test_network = split(content_path=content_path, rel_edges_fname=rel_edges_fname, id_field=id_field)
    train_df.to_csv(os.path.join(dataset_dir, "train.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(dataset_dir, "test.tsv"), sep="\t", index=False)
    write_edg_file(train_network, os.path.join(dataset_dir, "train_network.edg"))
    write_edg_file(test_network, os.path.join(dataset_dir, "test_network.edg"))
