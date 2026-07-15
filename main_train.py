import os.path
import time
import numpy as np
import pandas as pd
import yaml
from modelling.immense import train, train_w2v_model, get_or_create_bert_embeddings
from os import makedirs
from os.path import exists, join

from utils import load_from_pickle

np.random.seed(123)


if __name__ == "__main__":
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        train_dataset_params = params["train_dataset_params"]
        model_params = params["model_params"]
    field_id = dataset_general_params["field_id"]
    field_text = dataset_general_params["field_text"]
    field_label = dataset_general_params["field_label"]
    train_df = dataset_general_params["train_df"]
    embedding = dataset_general_params["embedding"]
    real_synthetic = dataset_general_params["real_synthetic"]

    path_rel = train_dataset_params["train_social_net"]
    consider_content = train_dataset_params["consider_content"]
    consider_rel = train_dataset_params["consider_rel"]
    consider_spat = train_dataset_params["consider_spat"]
    separator = train_dataset_params["separator"]
    retrain = train_dataset_params["retrain"]

    epochs_rel = model_params["epochs_rel"]
    mlp_batch_size = int(model_params["mlp_batch_size"])
    mlp_lr = float(model_params["mlp_lr"])
    ne_dim_rel = int(model_params["ne_dim_rel"])
    word_emb_size = int(model_params["word_emb_size"])
    w2v_epochs = int(model_params["w2v_epochs"])
    loss = model_params["loss"]
    models_dir = model_params["dir_models"]
    models_dir = os.path.join(models_dir, embedding, real_synthetic)

    if embedding=="bert":
        word_emb_size = 768
    elif embedding=="sonar":
        word_emb_size = 1024

    if not exists(models_dir):
        makedirs(models_dir)
    train_df = pd.read_csv(train_df, sep="\t")
    if embedding == "bert":
        users_embs_dict = get_or_create_bert_embeddings(train_df=train_df, model_dir=models_dir, id_field_name=field_id,
                                        text_field_name=field_text)
    elif embedding == "w2v":
        users_embs_dict = train_w2v_model(embedding_size=word_emb_size, epochs=w2v_epochs, id_field_name=field_id,
                                      model_dir=models_dir, text_field_name=field_text, train_df=train_df)
    elif embedding == "sonar":
        users_embs_dict = load_from_pickle(join(models_dir, f"{real_synthetic}_aggregated.pkl"))
        df_accounts = train_df["account_id"].unique().tolist()
        for k in list(users_embs_dict.keys()):
            if k not in df_accounts:
                users_embs_dict.pop(k)
        #train_df = train_df[train_df["account_id"].isin(list(users_embs_dict.keys()))]


    """confs = [(True, False, False), (True, False, True), (True, True, False), (True, True, True), (False, False, True), (False, True, False), (False, True, True)]
    for conf in confs:
        consider_content, consider_rel, consider_spat = conf[0], conf[1], conf[2]
        for loss in ["weighted", "focal"]:
            word_emb_size = 768
            for conf_ne in [(768, 768), (512, 512), (256, 256), (128, 128)]:
                ne_dim_rel = conf_ne[0]
                ne_dim_spat = conf_ne[1]"""
    print("CONTENT: {} REL: {} SPAT: {}".format(consider_content, consider_rel, consider_spat))
    now = time.time()
    train(train_df=train_df, model_dir=models_dir, gnn_batch_size=64, field_name_id=field_id,
          field_name_label=field_label, path_rel=path_rel, word_emb_size=word_emb_size,
          ne_dim_rel=ne_dim_rel, eps_nembs_rel=epochs_rel, consider_rel=consider_rel, separator=separator,
          consider_spat=consider_spat, embedding_type=embedding,
          consider_content=consider_content, users_embs_dict=users_embs_dict, loss=loss, retrain=retrain)
    print("Elapsed time: {}".format(time.time()-now))
