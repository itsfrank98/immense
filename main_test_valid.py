import time

import pandas as pd
import yaml
import os
from modelling.immense import test, get_or_create_bert_post_embeddings
from os.path import join, exists
from utils import load_from_pickle, read_df, _average_classification_reports, _save_classification_report_summaries, \
    aggregate_bert_post_embeddings
import numpy as np

def main_test():
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        test_dataset_params = params["test_dataset_params"]
        model_params = params["model_params"]

    field_id = dataset_general_params["field_id"]
    field_text = dataset_general_params["field_text"]
    field_label = dataset_general_params["field_label"]
    field_post_id = dataset_general_params["field_post_id"]
    all_posts_df_path = dataset_general_params["all_posts_with_users_path"]
    embedding = dataset_general_params["embedding"]
    aggr = dataset_general_params["aggr"]
    real_synthetic = dataset_general_params["real_synthetic"]

    consider_content = test_dataset_params["consider_content"]
    consider_rel = test_dataset_params["consider_rel"]
    separator = test_dataset_params["separator"]

    ne_dim_rel = int(model_params["ne_dim_rel"])
    loss = model_params["loss"]
    word_emb_size = int(model_params["word_emb_size"])
    use_ae = model_params["use_ae"]
    n_classes = model_params["out_channels"]

    percs = [2, 4, 8, 16, 32, 64, 75]       # percentage splits
    posts_df = pd.read_csv(all_posts_df_path, sep="\t")
    for p in percs:
        src = f"dataset/GAB_dataset/percented_train_{real_synthetic}/{p}_perc/cross_validation/"
        reports, binary_reports, maes, rmses = [], [], [], []
        reports_to_export = {}
        for fold in os.listdir(src):
            if os.path.isdir(os.path.join(src, fold)):
                working_dir = os.path.join(src, f"{fold}")
                train_df = read_df(join(working_dir, "train.tsv"))
                test_df = read_df(join(working_dir, "test.tsv"))
                path_rel = join(working_dir, "test_network.edg")
                mod_rel = softmax_model = w2v_model = dang_ae = safe_ae = None       # depending on the configuration, some of them will stay None, others won't

                if embedding == "bert":  #"sonar", "xlmt"]:
                    word_emb_size = 768
                    softmax_path = join(working_dir, f"softmax_model_{aggr}_{word_emb_size}_{loss}.pkl")  # _{embedding_type}
                    embeddings_path = join(f"dataset/GAB_dataset/percented_train_{real_synthetic}")
                    post_embeddings = get_or_create_bert_post_embeddings(model_dir=embeddings_path, text_field_name=field_text,
                                                                    post_id_field_name=field_post_id, posts_df=posts_df)
                    test_user_embeddings = aggregate_bert_post_embeddings(users_df=test_df, posts_embs_dict=post_embeddings,account_id_field=field_id,
                                                                          post_id_field=field_post_id, posts_df=posts_df)
                # TODO SISTEMARE ANCHE I CASI CON GLI ALTRI TIPI DI EMBEDDINGS. IL CARICAMENTO DEGLI USER TEXT EMBEDDINGS DEVE AVVENIRE QUI E NON NEL METODO TEST
                    if embedding == "bert":
                        word_emb_size = 768

                    else:
                        word_emb_size = 1024

                elif embedding == "w2v":
                    w2v_path = join(working_dir, "w2v_{}.pkl".format(word_emb_size))
                    w2v_model = load_from_pickle(w2v_path)

                if use_ae:
                    dang_ae = load_from_pickle(join(working_dir, f"autoencoderdang_{aggr}_{word_emb_size}.pkl"))
                    safe_ae = load_from_pickle(join(working_dir, f"autoencodersafe_{aggr}_{word_emb_size}.pkl"))
                elif exists(softmax_path):
                    print(f"[SOFTMAX] Loading softmax model from {softmax_path}")
                    softmax_model = load_from_pickle(softmax_path)
                    softmax_model.eval()
                else:
                    softmax_model = None

                mod_dir_rel = join(working_dir, "node_embeddings", "rel")
                mlp_name = "mlp"

                if consider_content:
                    mlp_name += "_content_{}".format(word_emb_size)
                if consider_rel:
                    mod_rel_name = f"graphsage_{aggr}_{ne_dim_rel}_{word_emb_size}_{loss}.pkl"
                    mod_rel = load_from_pickle(join(mod_dir_rel, mod_rel_name))
                    mlp_name += "_rel_{}".format(ne_dim_rel)
                mlp_name += f"_{loss}_{aggr}.pkl"
                mlp = load_from_pickle(join(working_dir, mlp_name))
                print(mlp_name.upper())
                print(f"LOSS: {loss}, REL DIM: {ne_dim_rel}, CONTENT DIM: {word_emb_size}, CONSIDER CONTENT: {consider_content},"
                      f" CONSIDER_REL: {consider_rel}, EMBEDDING TYPE: {embedding}, USE AE: {use_ae}")
                # Importante: questo codice non funzionerà se si passa un embedding diverso da bert, perchè la creazione del user embedding dict non è implementata per gli altri metodi
                report, binary_report, mae, rmse = test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, ae_risky=dang_ae,
                                ae_safe=safe_ae, full_embs_dict=test_user_embeddings, mod_rel=mod_rel, softmax_model=softmax_model,
                                use_ae=use_ae, rel_net_path=path_rel, field_name_text=field_text, models_dir=working_dir,
                                field_name_id=field_id, field_name_label=field_label, consider_content=consider_content, embedding_type=embedding,
                                consider_rel=consider_rel, separator=separator, real_synthetic="", aggr=aggr,
                                num_classes=n_classes, mlp=mlp)
                reports.append(report)
                binary_reports.append(binary_report)
                maes.append(mae)
                rmses.append(rmse)

        report_summary = _average_classification_reports(reports)
        reports_to_export["Multiclass"] = report_summary

        binary_report_summary = _average_classification_reports(binary_reports)
        reports_to_export["Binary"] = binary_report_summary

        _save_classification_report_summaries(
            reports_to_export,
            os.path.join(src, f"classification_report_summary.xlsx"),
            title=f"{src} folds",
            error_metrics={"Multiclass": {
                "MAE": {"mean": float(np.mean(maes)), "std": float(np.std(maes))},
                "RMSE": {"mean": float(np.mean(rmses)), "std": float(np.std(rmses))},
            }},
        )


if __name__ == "__main__":
    main_test()
