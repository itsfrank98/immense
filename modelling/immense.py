import os.path
import time
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, mean_absolute_error, root_mean_squared_error
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from triton.language import dtype

from modelling.softmax_classifier import SoftmaxClassifier
from modelling.ae import AE
from modelling.mlp import MLP
from modelling.sage import create_graph, create_mappers
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding_bert import WordEmb
from modelling.word_embedding_w2v import WordEmbW2V
from reduce_dimension import reduce_dimension
from utils import load_from_pickle, save_to_pickle, plot_confusion_matrix
np.random.seed(123)


def get_or_create_bert_embeddings(model_dir, train_df, id_field_name="id", text_field_name="text_cleaned",
                                  label_field_name="label"):
    """
    Se emb_file e lbl_file esistono -> li carica e restituisce.
    Altrimenti calcola embeddings con WordEmb, li salva e restituisce.
    """
    # raggruppa per id
    user_text_dict = train_df.groupby(id_field_name)[text_field_name].apply(
        lambda x: " ".join(x.astype(str).tolist())
    ).to_dict()
    embs_path = join(model_dir, "bert_embeddings.pt")
    if exists(embs_path):
        print(f"[BERT] Loading embeddings")
        emb_t = torch.load(embs_path)
        embeddings = emb_t.numpy() if isinstance(emb_t, torch.Tensor) else np.array(emb_t)
    else:
        start_emb = time.time()
        print(f"[BERT] File not found. Computing embeddings ...")
        # pulizia minima
        train_df = train_df.dropna(subset=[id_field_name, text_field_name, label_field_name])
        train_df = train_df[train_df[text_field_name].str.strip() != ""]
        if len(user_text_dict) == 0:
            raise RuntimeError("[BERT] No text in the dataset.")

        # embeddings con WordEmb
        emb_gen = WordEmb()
        embeddings = emb_gen.texts_to_vec(list(user_text_dict.values()))
        print(f"Embeddings learned and saved in {embs_path}. Elapsed time: {time.time() - start_emb:.1f}s")

        # this is to free up some space on the gpu
        emb_gen.model.to('cpu')
        del emb_gen
        torch.cuda.empty_cache()
    ids = list(user_text_dict.keys())
    embeddings_dict = {uid: emb for uid, emb in zip(ids, embeddings)}

    # salva come torch tensors
    torch.save(torch.tensor(embeddings), embs_path)

    return embeddings_dict


def train_w2v_model(embedding_size, epochs, id_field_name, model_dir, text_field_name, train_df):
    """
    Train the Word2Vc model that will be used for learning the embeddings of the content.
    :param embedding_size:
    :param epochs:
    :param id_field_name: Name of the field containing the ID in the training dataframe
    :param model_dir: Directory where the word embedding model will be saved
    :param text_field_name: Name of the field containing the text in the training dataframe
    :param train_df: training dataframe
    :return: dang_posts_array: Array of shape [n_dang_users, embedding_size] with the embeddings of the dangerous users
    :return: safe_posts_array: Array of shape [n_safe_users, embedding_size] with the embeddings of the safe users
    :return: users_embeddings (although the variable name is safe_users_embeddings): Dictionary having as keys the
        users' IDs and, for each of them, the embedding array given by the sum of the words in their posts
    """
    tok = TextPreprocessing()
    posts_content = tok.token_list(text_field_name=text_field_name, df=train_df)
    name = "w2v_{}.pkl".format(embedding_size)
    if not exists(join(model_dir, name)):
        # tracker = EmissionsTracker()
        # tracker.start()
        start_emb = time.time()
        print("Training word2vec model")
        w2v_model = WordEmbW2V(posts_content, embedding_size=embedding_size, window=10, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        # tracker.stop()
        save_to_pickle(join(model_dir, name), w2v_model)
        print("Elapsed time for training {} w2v: {}".format(embedding_size, time.time() - start_emb))

    else:
        print("Loading word2vec model")
        w2v_model = load_from_pickle(join(model_dir, name))
    # split content in safe and dangerous
    all_users_tokens = tok.token_dict(train_df, text_field_name=text_field_name, id_field_name=id_field_name)
    all_users_embeddings = w2v_model.text_to_vec(users=all_users_tokens)  # Get a dict of all the embeddings of each user, keeping the association with the key
    print(len(all_users_embeddings))
    # for k in list(all_users_embeddings.keys()):
    #     if type(all_users_embeddings[k]) != np.ndarray:
    #         all_users_embeddings.pop(k)
    return all_users_embeddings


def model_fusion(model_dir, y_train, content_embs, mlp_name, mlp_loss, mlp_lr=.004, ae_dang=None, ae_safe=None,
                 rel_preds=None, spat_preds=None, weights=None):
    """
    Train the MLP aimed at fusing the models
    Args:
    :param model_dir:
    :param y_train: Train labels
    :param content_embs: torch tensor containing the word embeddings of the content posted by the users. The features
    are z-score normalized
    :param mlp_lr: learning rate for the mlp
    :param ae_dang: Dangerous autoencoder model. Leave it to None if you don't want to consider the content analysis module
    :param ae_safe: Safe autoencoder model. Leave it to None if you don't want to consider the content analysis module
    :param rel_preds: Predictions from the relational node embedding module. Leave it to none if you don't want to use that module
    :param spat_preds: Predictions from the spatial node embedding module. Leave it to none if you don't want to use that module
    :param mlp_name: File name of the mlp to create
    Returns: The learned MLP
    """
    n_columns = 5
    print(f"N OF COLUMNS {n_columns}")
    dataset = torch.zeros((content_embs.shape[0], n_columns))

    if ae_dang and ae_safe:
        prediction_safe = ae_safe.predict(content_embs)
        prediction_risky = ae_dang.predict(content_embs)
        mse_loss = MSELoss()
        prediction_loss_safe = []
        prediction_loss_risky = []
        for i in range(content_embs.shape[0]):
            prediction_loss_safe.append(mse_loss(content_embs[i], prediction_safe[i]))
            prediction_loss_risky.append(mse_loss(content_embs[i], prediction_risky[i]))
        labels = [0 if i < j else 1 for i, j in zip(prediction_loss_safe, prediction_loss_risky)]
        dataset[:, 0] = torch.tensor(prediction_loss_safe, dtype=torch.float32)
        dataset[:, 1] = torch.tensor(prediction_loss_risky, dtype=torch.float32)
        dataset[:, 2] = torch.tensor(labels, dtype=torch.float32)

    if rel_preds is not None:
        safe_rel_probs = torch.tensor(rel_preds[:, 0], dtype=torch.float32)
        risky_rel_probs = torch.tensor(rel_preds[:, 1], dtype=torch.float32)
        dataset[:, 3], dataset[:, 4] = safe_rel_probs, risky_rel_probs
    if spat_preds is not None:
        safe_spat_probs = torch.tensor(spat_preds[:, 0], dtype=torch.float32)
        risky_spat_probs = torch.tensor(spat_preds[:, 1], dtype=torch.float32)
        dataset[:, 5], dataset[:, 6] = safe_spat_probs, risky_spat_probs

    mlp = MLP(X_train=dataset, y_train=y_train, model_path=join(model_dir, mlp_name), weights=weights, loss=mlp_loss)
    optim = Adam(mlp.parameters(), lr=mlp_lr, weight_decay=1e-4)
    mlp.train_mlp(optim)


def model_fusion_softmax(model, model_dir, y_train, content_embs, mlp_name, mlp_loss, out_channels,
                         mlp_lr=.004, rel_preds=None, spat_preds=None, weights=None):
    """
    Variante di model_fusion per l'uso con SoftmaxClassifier.
    Combina le predizioni del classificatore Softmax con le feature relazionali/spaziali
    per addestrare l'MLP finale.
    Args:
        model: il classificatore Softmax già addestrato/caricato
        model_dir: directory dove salvare il modello MLP
        y_train: etichette di training
        content_embs: embeddings del contenuto (torch tensor)
        mlp_name: nome file per salvare l'MLP
        mlp_loss: tipo di loss da usare nell'MLP
        mlp_lr: learning rate per l'MLP
        rel_preds: predizioni del modulo relazionale (2 colonne)
        spat_preds: predizioni del modulo spaziale (2 colonne)
        weights: pesi per classi (opzionali)
        out_channels: number of output channels for the final MLP
    """
    # 6 colonne: 2 softmax + eventuali 2 rel + eventuali 2 spat
    n_columns = out_channels * 2
    print(f"N OF COLUMNS {n_columns}")
    n_rows = content_embs.shape[0]

    dataset = torch.zeros((n_rows, n_columns))

    softmax_outputs = torch.tensor(model.predict_proba(content_embs), dtype=torch.float32)
    dataset[:, 0:out_channels] = softmax_outputs

    if rel_preds is not None:
        dataset[:, out_channels:2*out_channels] = torch.tensor(rel_preds[:, 0:out_channels], dtype=torch.float32)

    # if spat_preds is not None:
    #     dataset[:, 4:6] = torch.tensor(spat_preds[:, 0:2], dtype=torch.float32)


    # === Addestramento MLP finale ===
    mlp = MLP(
        X_train=dataset,
        y_train=y_train,
        model_path=join(model_dir, mlp_name),
        weights=weights,
        loss=mlp_loss,
        num_channels=out_channels
    )
    optim = Adam(mlp.parameters(), lr=mlp_lr, weight_decay=1e-4)
    mlp.train_mlp(optim)

    return mlp


def get_relational_preds(df, node_embs=None):
    dataset = torch.zeros(len(df), 2)
    dataset[:, 0] = torch.tensor(node_embs[:, 0], dtype=torch.float32)
    dataset[:, 1] = torch.tensor(node_embs[:, 1], dtype=torch.float32)
    return dataset


def train(field_name_id, field_name_label, model_dir, train_df, word_emb_size, users_embs_dict, separator, loss, use_ae,
          embedding_type, aggregation, out_channels, gnn_batch_size=None, consider_content=True, consider_rel=True,
          consider_spat=True, ne_dim_rel=None, ne_dim_spat=None, eps_nembs_rel=None, eps_nembs_spat=None, path_rel=None,
          path_spat=None, retrain=False, softmax_lr=None):
    """
    Builds and trains the independent modules that analyze content, social relationships and spatial relationships, and
    then fuses them with the MLP
    :param field_name_id: Name of the field containing the id
    :param field_name_label: Name of the field containing the label
    :param model_dir: Directory where the models will be saved
    :param ne_dim_rel: Dimension of the relational node embeddings to learn
    :param ne_dim_spat: Dimension of the spatial node embeddings to learn
    :param train_df: Dataframe with the posts used for the MLP training
    :param word_emb_size: Dimension of the word embeddings to create
    :param users_embs_dict: Dictionary having as keys the user IDs and as values their associated word embedding vector
    :param separator: Separator used in the edgelist
    :param gnn_batch_size: Batch size for learning node embedding models
    :param consider_content: Whether to use the module for the semantic content analysis
    :param consider_rel: Whether to use the module for the analysis of social relationships
    :param consider_spat: Whether to use the module for the analysis of spatial relationships
    :param eps_nembs_rel: Epochs for training the relational node embedding model. Can be None if consider_rel=False
    :param eps_nembs_spat: Epochs for training the spatial node embedding model. Can be None if consider_spat=False
    :param path_rel: Path to the file stating the social relationships among the users. Can be None if consider_rel=False
    :param path_spat: Path to the file stating the spatial relationships among the users. Can be None if consider_spat=False
    :param embedding_type: set this to true for using bert embeddings, to false for using word2vec embeddings
    :return: Nothing, the learned mlp will be saved in the file "mlp.h5" and put in the model directory
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = list(users_embs_dict.keys())
    train_df = train_df[train_df[field_name_id].isin(ids)]
    y_train = list(train_df[field_name_label])
    dang_posts_ids = list(train_df.loc[train_df[field_name_label] == 1][field_name_id])
    safe_posts_ids = list(train_df.loc[train_df[field_name_label] == 0][field_name_id])

    keys = list(users_embs_dict.keys())

    dang_users_ar = np.array([users_embs_dict[k] for k in keys if k in dang_posts_ids]) if len(keys) > 0 else np.array([])
    safe_users_ar = np.array([users_embs_dict[k] for k in keys if k in safe_posts_ids]) if len(keys) > 0 else np.array([])

    weights = None
    if loss == "weighted":
        nz = len(train_df[train_df[field_name_label] == 1])
        if nz == 0:
            raise RuntimeError("Nessun esempio della classe positiva trovato nel train_df.")
        pos_weight = len(train_df) / nz
        neg_weight = len(train_df) / (len(train_df) - nz)
        #neg_weight = len(train_df) / (2 * (len(train_df) - nz))
        weights = torch.tensor([neg_weight, pos_weight])
        weights = weights.to(device)

    mlp_name = "mlp"
    safe_ae = risky_ae = None
    posts_embs = None
    # if consider_content:
    #     mlp_name += "_content_{}".format(word_emb_size)
    #     if embedding_type=="w2v" or use_ae:
    #         posts_embs = torch.tensor(np.array(list(users_embs_dict.values())), dtype=torch.float32)
    #         safe_ae_name = join(model_dir, f"autoencodersafe_{aggregation}_{word_emb_size}.pkl")
    #         risky_ae_name = join(model_dir, f"autoencoderdang_{aggregation}_{word_emb_size}.pkl")
    #
    #         if not exists(safe_ae_name):  # or retrain:
    #             safe_ae = AE(X_train=safe_users_ar, epochs=75, batch_size=64, lr=0.002, name=safe_ae_name)
    #             safe_ae.train_autoencoder_content()
    #         else:
    #             safe_ae = load_from_pickle(safe_ae_name)
    #
    #         if not exists(risky_ae_name):  # or retrain:
    #             risky_ae = AE(X_train=dang_users_ar, epochs=100, batch_size=32, lr=0.002, name=risky_ae_name)
    #             risky_ae.train_autoencoder_content()
    #         else:
    #             risky_ae = load_from_pickle(risky_ae_name)
    #     else:
    #         softmax_pkl_path = join(model_dir, f"softmax_model_{aggregation}_{word_emb_size}_{loss}.pkl")
    #         softmax_state_path = join(model_dir, f"softmax_model_state_{aggregation}_{word_emb_size}_{loss}.pt")
    #         # labels allineati agli ids
    #         ids = train_df[field_name_id].tolist()
    #         labels_map = train_df.drop_duplicates(field_name_id).set_index(field_name_id)[field_name_label].to_dict()
    #         labels = np.array([labels_map[uid] for uid in ids])
    #         y_train = torch.tensor(labels, dtype=torch.float32)
    #         posts_embs = torch.tensor(np.array(list(users_embs_dict.values())), dtype=torch.float32)
    #
    #         if (not exists(softmax_pkl_path) and not exists(softmax_state_path)) or retrain:
    #             train_ds = TensorDataset(posts_embs, y_train)
    #             train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    #
    #             model = SoftmaxClassifier(
    #                 embedding_dim=posts_embs.shape[1],
    #                 num_classes=int(len(np.unique(labels))),
    #                 loss=loss,
    #                 weights=weights
    #             )
    #
    #             makedirs(model_dir, exist_ok=True)
    #             model.train_model(train_loader, num_epochs=10, learning_rate=softmax_lr,
    #                               state_path=softmax_state_path, pkl_path=softmax_pkl_path)
    #         else:
    #             if exists(softmax_pkl_path):
    #                 model = load_from_pickle(softmax_pkl_path)
    #             else:
    #                 model = SoftmaxClassifier(
    #                     embedding_dim=posts_embs.shape[1],
    #                     num_classes=int(len(np.unique(labels))),
    #                     loss=loss,
    #                     weights=weights
    #                 )
    #                 state = torch.load(softmax_state_path, map_location=torch.device("cpu"))
    #                 model.load_state_dict(state)

    model_dir_rel = join(model_dir, "node_embeddings", "rel")
    model_dir_spat = join(model_dir, "node_embeddings", "spat")
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass

    x_rel = x_spat = None
    torch.cuda.empty_cache()

    if consider_rel:
        mlp_name += f"_rel_{ne_dim_rel}"
        x_rel = reduce_dimension(model_dir=model_dir_rel, edge_path=path_rel, lab="rel", ne_dim=ne_dim_rel,
                                 train_df=train_df, epochs=eps_nembs_rel, sizes=[2, 3], loss=loss,
                                 features_dict=users_embs_dict, batch_size=gnn_batch_size, training_weights=weights,
                                 we_dim=word_emb_size, retrain=retrain, separator=separator, aggregation=aggregation,
                                 field_name_id=field_name_id, field_name_label=field_name_label, out_channels=out_channels)
        torch.cuda.empty_cache()
    # if consider_spat:
    #     mlp_name += f"_spat_{ne_dim_rel}"
    #     x_spat = reduce_dimension(model_dir=model_dir_spat, edge_path=path_spat, lab="spat", ne_dim=ne_dim_spat,
    #                               train_df=train_df, epochs=eps_nembs_spat, sizes=[3, 5],
    #                               features_dict=users_embs_dict, batch_size=gnn_batch_size, aggregation=aggregation,
    #                               training_weights=weights, we_dim=word_emb_size, retrain=retrain, separator=separator,
    #                               field_name_id=field_name_id, field_name_label=field_name_label, loss=loss, out_channels=out_channels)
    #     torch.cuda.empty_cache()

    # mlp_name += f"_{loss}_{aggregation}.pkl"
    # print("Learning MLP...\n")
    #
    # if use_ae:
    #     model_fusion(ae_dang=risky_ae, ae_safe=safe_ae, content_embs=posts_embs, model_dir=model_dir, rel_preds=x_rel,
    #                  spat_preds=x_spat, y_train=y_train, weights=weights, mlp_name=mlp_name, mlp_loss=loss)
    # else:
    #     mlp = model_fusion_softmax(model=model, model_dir=model_dir, y_train=y_train, content_embs=posts_embs,
    #                                mlp_name=mlp_name, mlp_loss=loss, rel_preds=x_rel, spat_preds=x_spat,
    #                                weights=weights, out_channels=out_channels)
    #     if mlp is not None:
    #         save_to_pickle(join(model_dir, mlp_name), mlp)


def test(test_df, train_df, field_name_id, field_name_text, field_name_label, mlp: MLP, w2v_model, consider_content, use_ae,
         consider_rel, consider_spat, models_dir, real_synthetic, num_classes, embedding_type, ae_risky=None, ae_safe=None, aggr="sum",
         mod_rel=None, mod_spat=None, rel_net_path=None, softmax_model=None, spat_net_path=None, separator="\t"):
    tok = TextPreprocessing()
    posts_test = tok.token_dict(test_df, text_field_name=field_name_text, id_field_name=field_name_id)

    if use_ae:
        n_of_columns = 1 + num_classes * 3
    if embedding_type == "xlmt":
        n_of_columns = num_classes * 2  # we have 6 outputs for the content analysis part and 6 for the social network. Spatial part is completely removed, not even zeroed
    else:
        n_of_columns = num_classes * 3
    if embedding_type == "bert":
        users_embs_dict_test = get_or_create_bert_embeddings(model_dir=models_dir, train_df=test_df, id_field_name=field_name_id,
                                                  text_field_name=field_name_text, label_field_name=field_name_label)
    elif embedding_type == "w2v":
        posts_train = tok.token_dict(test_df, text_field_name=field_name_text, id_field_name=field_name_id)
        full_embs_dict = w2v_model.text_to_vec(posts_train)
        users_embs_dict_test = w2v_model.text_to_vec(posts_test)
        full_embs_dict.update(users_embs_dict_test)
    elif embedding_type == "sonar":
        full_embs_dict = load_from_pickle(os.path.join(models_dir, f"{real_synthetic}_aggregated_{aggr}.pkl"))
        full_embs_dict = {int(k): v for k, v in full_embs_dict.items()}
        df_accounts = test_df["account_id"].unique().tolist()
        users_embs_dict_test = full_embs_dict.copy()      # full_embs_dict will be used to create the full graph. embs_dict contains only the test set user embeddings
        for k in list(users_embs_dict_test.keys()):
            if k not in df_accounts:
                users_embs_dict_test.pop(k)
        test_df = test_df[test_df["account_id"].isin(list(users_embs_dict_test.keys()))]
    elif embedding_type == "xlmt":
        data = torch.load(join(models_dir, "test_user_embeddings.pt"))
        ids, emb = list(data.values())
        full_embs_dict = dict(zip(ids, emb.unbind(0)))
    ids = list(full_embs_dict.keys())
    test_df = test_df[test_df[field_name_id].isin(ids)]
    test_set = torch.zeros(len(test_df), n_of_columns)

    # if consider_content:
    #     posts_embs = torch.tensor(np.array(list(full_embs_dict.values())), dtype=torch.float32)
    #     if use_ae:
    #         pred_safe = ae_safe.predict(posts_embs)
    #         pred_risky = ae_risky.predict(posts_embs)
    #         loss = MSELoss()
    #         pred_loss_safe = []
    #         pred_loss_risky = []
    #         for i in range(posts_embs.shape[0]):
    #             pred_loss_safe.append(loss(posts_embs[i], pred_safe[i]))
    #             pred_loss_risky.append(loss(posts_embs[i], pred_risky[i]))
    #
    #         labels = [0 if i < j else 1 for i, j in zip(pred_loss_safe, pred_loss_risky)]
    #         test_set[:, 0] = torch.tensor(pred_loss_safe, dtype=torch.float32)
    #         test_set[:, 1] = torch.tensor(pred_loss_risky, dtype=torch.float32)
    #         test_set[:, 2] = torch.tensor(labels, dtype=torch.float32)
    #     else:
    #         content_probs = softmax_model.predict_proba(posts_embs)
    #         test_set[:, 0:num_classes] = torch.Tensor(content_probs)
    #         logits = softmax_model(posts_embs)
    #         content_probs = torch.softmax(logits, dim=1).cpu()
    #         test_set[:, 0:2] = content_probs

    if consider_rel:
        mapper, inv_map_rel = create_mappers(full_embs_dict)
        graph = create_graph(inv_map=inv_map_rel, weighted=False, features=full_embs_dict, edg_dir=rel_net_path,
                             df=pd.concat([test_df, train_df]).drop_duplicates(subset=field_name_id),
                             separator=separator, field_name_id=field_name_id, field_name_label=field_name_label)
        with torch.no_grad():
            graph = graph.to(mod_rel.device)
            rel_preds_whole_graph = mod_rel(graph, inference=True).cpu()
        test_ids = test_df["account_id"].unique().tolist()
        test_rows = [inv_map_rel[uid] for uid in test_ids]
        rel_preds = rel_preds_whole_graph[test_rows]
        #rel_preds = torch.from_numpy(rel_preds, type=torch.float32)
        if embedding_type != "xlmt":
            safe_rel_probs = rel_preds[:, 0]
            risky_rel_probs = rel_preds[:, 1]
            index = 3 if use_ae else 2
            test_set[:, index], test_set[:, index+1] = safe_rel_probs, risky_rel_probs
        else:
            test_set[:, num_classes:n_of_columns] = rel_preds

    # if consider_spat:
    #     mapper, inv_map_sp = create_mappers(users_embs_dict_test)
    #     graph = create_graph(inv_map=inv_map_sp, weighted=True, features=users_embs_dict_test, edg_dir=spat_net_path, df=test_df,
    #                          separator=separator, field_name_id=field_name_id, field_name_label=field_name_label)
    #     with torch.no_grad():
    #         graph = graph.to(mod_spat.device)
    #         spat_preds = mod_spat(graph, inference=False).cpu().detach().numpy()
    #     safe_spat_probs = torch.tensor(spat_preds[:, 0], dtype=torch.float32)
    #     risky_spat_probs = torch.tensor(spat_preds[:, 1], dtype=torch.float32)
    #     index = 4 if embedding_type else 5
    #     test_set[:, index], test_set[:, index+1] = safe_spat_probs, risky_spat_probs

    #pred = mlp.test(test_set)
    pred = rel_preds.argmax(dim=1)
    ts = test_set.detach().numpy()
    y_true = np.array(test_df[field_name_label])

    report = classification_report(y_true=y_true, y_pred=pred, output_dict=True, zero_division=0)
    print(report)

    binary_pred = [0 if p<3 else 1 for p in pred]
    y_true_pred = [0 if p<3 else 1 for p in y_true]
    binary_report = classification_report(y_true=y_true_pred, y_pred=binary_pred, output_dict=True, zero_division=0)
    mae = mean_absolute_error(y_true=y_true, y_pred=pred)
    rmse = root_mean_squared_error(y_true=y_true, y_pred=pred)
    print(binary_report)
    """
    pred_rel = np.argmax(rel_preds, 1)
    print("RELATIONAL")
    print(classification_report(y_true=y_true, y_pred=pred_rel))


    print("CONTENT")
    pred_con = np.argmax(content_probs, 1)
    print(classification_report(y_true=y_true, y_pred=pred_con))
    #"""
    return report, binary_report, mae, rmse