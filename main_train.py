import time
import numpy as np
import pandas as pd
import yaml

from modelling.immense import train, train_w2v_model, get_or_create_bert_post_embeddings
from utils import aggregate_bert_post_embeddings
from os import makedirs
from os.path import exists, join


from utils import save_to_pickle, load_from_pickle

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
    posts_df_path = dataset_general_params["posts_df"]
    bert = dataset_general_params["bert"]

    path_rel = train_dataset_params["train_social_net"]
    path_spat = train_dataset_params["train_spatial_net"]
    consider_content = train_dataset_params["consider_content"]
    consider_rel = train_dataset_params["consider_rel"]
    consider_spat = train_dataset_params["consider_spat"]
    separator = train_dataset_params["separator"]
    retrain = train_dataset_params["retrain"]

    epochs_rel = model_params["epochs_rel"]
    epochs_spat = model_params["epochs_spat"]
    mlp_batch_size = int(model_params["mlp_batch_size"])
    mlp_lr = float(model_params["mlp_lr"])
    ne_dim_rel = int(model_params["ne_dim_rel"])
    ne_dim_spat = int(model_params["ne_dim_spat"])
    word_emb_size = int(model_params["word_emb_size"])
    w2v_epochs = int(model_params["w2v_epochs"])
    loss = model_params["loss"]
    models_dir = model_params["dir_models"]
    if bert:
        models_dir += "/with_bert"
        word_emb_size = 768
    else:
        models_dir += "/with_w2v"


    if not exists(models_dir):
        makedirs(models_dir)
    train_df = pd.read_csv(train_df)
    if bert:
        posts_df = pd.read_csv(posts_df_path)
        users_embs_pah = join(models_dir, "users_bert_embs_dict_avg.pkl")
        if not exists(users_embs_pah):
            posts_embs_dict = get_or_create_bert_post_embeddings(model_dir=models_dir, text_field_name=field_text, posts_df=posts_df)
            users_embs_dict = aggregate_bert_post_embeddings(users_df=train_df, posts_embs_dict=posts_embs_dict, posts_df=posts_df)
            save_to_pickle(users_embs_pah, users_embs_dict)
        else:
            users_embs_dict = load_from_pickle(users_embs_pah)
    else:
        users_embs_dict = train_w2v_model(embedding_size=word_emb_size, epochs=w2v_epochs, id_field_name=field_id,
                                      model_dir=models_dir, text_field_name=field_text, train_df=train_df)

    co = 1
    confs = [(True, False, False), (True, False, True), (True, True, False), (True, True, True), (False, False, True), (False, True, False), (False, True, True)]
    for loss in ["weighted", "focal"]:
        for conf in confs:
            consider_content, consider_rel, consider_spat = conf[0], conf[1], conf[2]
            word_emb_size = 768
            for conf_ne in [(768, 768), (512, 512), (256, 256), (128, 128)]:
                ne_dim_rel = conf_ne[0]
                ne_dim_spat = conf_ne[1]
                print("\n")
                print(f"{co}/{len(loss)*len(conf)*len(conf_ne)}")
                print("CONTENT: {} REL: {} SPAT: {}".format(consider_content, consider_rel, consider_spat))
                now = time.time()
                train(train_df=train_df, model_dir=models_dir, gnn_batch_size=64, field_name_id=field_id,
                      field_name_label=field_label, path_rel=path_rel, path_spat=path_spat, word_emb_size=word_emb_size,
                      ne_dim_spat=ne_dim_spat, ne_dim_rel=ne_dim_rel, eps_nembs_spat=epochs_spat, eps_nembs_rel=epochs_rel,
                      consider_rel=consider_rel, separator=separator, consider_spat=consider_spat, bert=bert,
                      consider_content=consider_content, users_embs_dict=users_embs_dict, loss=loss, retrain=retrain)
                print("Elapsed time: {}".format(time.time()-now))
        co += 1