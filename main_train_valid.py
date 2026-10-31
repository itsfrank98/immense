import os.path
import time
import numpy as np
import pandas as pd
import yaml
from modelling.immense import train
from os import makedirs
from os.path import exists, join
import torch

np.random.seed(123)


if __name__ == "__main__":
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        train_dataset_params = params["train_dataset_params"]
        model_params = params["model_params"]

    real_synthetic = dataset_general_params["real_synthetic"]
    field_id = dataset_general_params["field_id"]
    field_text = dataset_general_params["field_text"]
    field_label = dataset_general_params["field_label"]
    train_df = dataset_general_params["train_df"]
    embedding = dataset_general_params["embedding"]
    aggr = dataset_general_params["aggr"]

    consider_content = train_dataset_params["consider_content"]
    consider_rel = train_dataset_params["consider_rel"]
    consider_spat = train_dataset_params["consider_spat"]
    separator = train_dataset_params["separator"]
    retrain = train_dataset_params["retrain"]

    epochs_rel = model_params["epochs_rel"]
    mlp_batch_size = int(model_params["mlp_batch_size"])
    softmax_lr = float(model_params["softmax_lr"])
    mlp_lr = float(model_params["mlp_lr"])
    ne_dim_rel = int(model_params["ne_dim_rel"])
    loss = model_params["loss"]
    use_ae = model_params["use_ae"]
    out_channels = model_params["out_channels"]

    if embedding=="bert" or "xlmt":
        word_emb_size = 768
    elif embedding=="sonar":
        word_emb_size = 1024

    percs = [1, 2, 4, 8, 16, 32, 64, 75]

    for p in percs:
        src = f"dataset/GAB_dataset/percented_train_{real_synthetic}/{p}_perc/cross_validation/"
        print(src)
        for fold in os.listdir(src):
            if os.path.isdir(os.path.join(src, fold)):
                working_dir = os.path.join(src, f"{fold}")
                path_rel = os.path.join(working_dir, "train_network.edg")
                train_df = pd.read_csv(os.path.join(working_dir, "train.tsv"), sep="\t")

                data = torch.load(join(working_dir, "train_user_embeddings.pt"))
                ids, emb = list(data.values())
                users_embs_dict = dict(zip(ids, emb.unbind(0)))
                print(f"PERC: {p}, FOLD: {fold}, LOSS: {loss}, REL DIM: {ne_dim_rel}, CONTENT DIM: {word_emb_size}, "
                      f"CONSIDER CONTENT: {consider_content}, CONSIDER_REL: {consider_rel}, EMBEDDING TYPE: {embedding}, ")
                now = time.time()
                train(train_df=train_df, model_dir=working_dir, gnn_batch_size=64, field_name_id=field_id,
                      field_name_label=field_label, path_rel=path_rel, word_emb_size=word_emb_size,
                      ne_dim_rel=ne_dim_rel, eps_nembs_rel=epochs_rel, consider_rel=consider_rel, separator=separator,
                      consider_spat=consider_spat, embedding_type=embedding, consider_content=consider_content, aggregation=aggr,
                      users_embs_dict=users_embs_dict, loss=loss, retrain=retrain, softmax_lr=softmax_lr, use_ae=use_ae,
                      out_channels=out_channels)
                print("Elapsed time: {}".format(time.time()-now))
