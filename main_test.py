import time
import pandas as pd
import yaml
import os
from modelling.immense import test
from os.path import join, exists
from utils import load_from_pickle, read_df


def main_test():
    now = time.time()
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        test_dataset_params = params["test_dataset_params"]
        model_params = params["model_params"]

    train_df = dataset_general_params["train_df"]
    field_id = dataset_general_params["field_id"]
    field_text = dataset_general_params["field_text"]
    field_label = dataset_general_params["field_label"]
    embedding = dataset_general_params["embedding"]
    real_synthetic = dataset_general_params["real_synthetic"]

    test_df = test_dataset_params["test_df"]
    path_rel = test_dataset_params["full_social_net"]
    consider_content = test_dataset_params["consider_content"]
    consider_rel = test_dataset_params["consider_rel"]
    consider_spat = test_dataset_params["consider_spat"]
    separator = test_dataset_params["separator"]

    ne_dim_rel = int(model_params["ne_dim_rel"])
    ne_dim_spat = int(model_params["ne_dim_spat"])
    loss = model_params["loss"]
    word_emb_size = int(model_params["word_emb_size"])
    models_dir = join(model_params["dir_models"], embedding)
    path_spat = None

    train_df = read_df(train_df)
    test_df = read_df(test_df)
    mod_rel = mod_spat = softmax_model = w2v_model = dang_ae = safe_ae = None       # depending on the configuration, some of them will stay None, others won't

    if embedding in ["bert", "sonar"]:
        if embedding == "bert":
            word_emb_size = 768
        else:
            word_emb_size = 1024
        softmax_path = join(models_dir, f"softmax_model_{embedding}_{loss}.pkl")   # _{embedding_type}

        if exists(softmax_path):
            print(f"[SOFTMAX] Loading softmax model from {softmax_path}")
            softmax_model = load_from_pickle(softmax_path)
            softmax_model.eval()
        else:
            raise FileNotFoundError(f"[ERROR!] Softmax model not found in {softmax_path}")
    elif embedding == "w2v":
        w2v_path = join(models_dir, "w2v_{}.pkl".format(word_emb_size))
        w2v_model = load_from_pickle(w2v_path)
        dang_ae = load_from_pickle(join(models_dir, "autoencoderdang_{}.pkl".format(word_emb_size)))
        safe_ae = load_from_pickle(join(models_dir, "autoencodersafe_{}.pkl".format(word_emb_size)))

    mod_dir_rel = join(models_dir, "node_embeddings", "rel")
    mod_dir_spat = join(models_dir, "node_embeddings", "spat")
    mlp_name = "mlp"

    if consider_content:
        mlp_name += "_content_{}".format(word_emb_size)
    if consider_rel:
        mod_rel_name = f"graphsage_{ne_dim_rel}_{word_emb_size}_{loss}.pkl"
        mod_rel = load_from_pickle(join(mod_dir_rel, mod_rel_name))
        mlp_name += "_rel_{}".format(ne_dim_rel)
    if consider_spat:
        mod_spat_name = f"graphsage_{ne_dim_spat}_{word_emb_size}_{loss}.pkl"
        mod_spat = load_from_pickle(join(mod_dir_spat, mod_spat_name))
        mlp_name += "_spat_{}".format(ne_dim_spat)
    mlp_name += "_{}.pkl".format(loss)
    mlp = load_from_pickle(join(models_dir, "mlp", mlp_name))
    print(mlp_name.upper())
    test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, ae_risky=dang_ae, ae_safe=safe_ae, mlp=mlp, mod_rel=mod_rel, softmax_model=softmax_model,
         mod_spat=mod_spat, rel_net_path=path_rel, spat_net_path=path_spat, field_name_text=field_text, models_dir=models_dir,
         field_name_id=field_id, field_name_label=field_label, consider_content=consider_content, embedding_type=embedding,
         consider_rel=consider_rel, consider_spat=consider_spat, separator=separator, mlp_loss=loss, real_synthetic=real_synthetic)
    print(f"ELAPSED TIME: {time.time()-now}")


if __name__ == "__main__":
    main_test()
