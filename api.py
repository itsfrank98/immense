from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse, abort
from modelling.sairus import classify_users
from task_manager.tasks import train_task, preprocess_task, CONTENT_FILENAME, JOBS_DIR, ID2IDX_REL_FILENAME, ID2IDX_SPAT_FILENAME, REL_ADJ_MAT_FILENAME, SPAT_ADJ_MAT_FILENAME
from os.path import exists, join

api = Api(title="SNA spatial and textual API", version="0.1", description="Social Network Analysis API with spatial and textual information")
application = Flask(__name__)
api.init_app(application)
preprocess_parser = reqparse.RequestParser()
preprocess_parser.add_argument('content_url', type=str, required=True, help="Path to the file containing the unprocessed content")
preprocess_parser.add_argument('rel_url', type=str, help="Path to the file containing the social relationships among users")
preprocess_parser.add_argument('id_field_name', type=str, help="Name of the field containing the user ids")
preprocess_parser.add_argument('text_field_name', type=str, help="Name of the field containing the text")
@api.route("/node_classification/preprocess", methods=['POST'])
class Preprocess(Resource):
    """Preprocess the posts, concatenate and aggregate them"""
    @api.expect(preprocess_parser)
    def post(self):
        params = preprocess_parser.parse_args()
        content_url = params['content_url']
        rel_url = params['rel_url']
        id_field_name = params['id_field_name']
        text_field_name = params['text_field_name']
        task = preprocess_task.apply_async(args=[content_url, rel_url, id_field_name, text_field_name])
        return jsonify({"Job id": task.id})
        #preprocess_task(content_url=content_url, rel_url=rel_url, id_field_name=id_field_name, text_field_name=text_field_name)


train_parser = reqparse.RequestParser()
train_parser.add_argument('job_id', type=str, required=True, help="ID of the job that created the dataset")
train_parser.add_argument('rel_ne_tec', type=str, required=True, default="node2vec", choices=("none", "autoencoder", "node2vec", "pca"),
                          help="Technique for learning relational node embeddings. Must be one of the following:"
                               "'ae' (autoencoder); 'none' (directly uses the adj matrix rows); 'node2vec'; 'pca'")
train_parser.add_argument('spat_ne_tec', type=str, required=True, default="node2vec", choices=("none", "autoencoder", "node2vec", "pca"),
                          help="Technique for learning spatial node embeddings. Must be one of the following:"
                               "'ae' (autoencoder); 'none' (directly uses the adj matrix rows); 'node2vec'; 'pca'")
train_parser.add_argument('kc', type=int, required=True, default=128, help="Dimension of the word embeddings")
train_parser.add_argument('window', type=int, default=5, required=True, help="Size of the window used to learn word embeddings")
train_parser.add_argument('w2v_epochs', type=int, default=1, required=True, help="No. epochs for training the word embedding model")
train_parser.add_argument('kr', type=int, default=128, required=True, help="Dimension of relational node embeddings to learn")
train_parser.add_argument('ks', type=int, default=128, required=True, help="Dimension of spatial node embeddings to learn")
train_parser.add_argument('n_of_walks_rel', type=int, default=10, required=False, help="Number of walks for the relational n2v embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('n_of_walks_spat', type=int, default=10, required=False, help="Number of walks for the spatial n2v embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('walk_length_rel', type=int, default=5, required=False, help="Walk length for the relational n2v embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('walk_length_spat', type=int, default=5, required=False, help="Walk length for the spatial n2v embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('p_rel', type=int, default=1, required=False, help="Node2vec's hyperparameter p for the relational embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('p_spat', type=int, default=1, required=False, help="Node2vec's hyperparameter p for the spatial embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('q_rel', type=int, default=4, required=False, help="Node2vec's hyperparameter q for the relational embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('q_spat', type=int, default=4, required=False, help="Node2vec's hyperparameter q for the spatial embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('n2v_epochs_rel', type=int, default=1, required=False, help="No. epochs for training the relational n2v embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('n2v_epochs_spat', type=int, default=1, required=False, help="No. epochs for training the spatial n2v embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('rel_autoencoder_epochs', type=int, default=100, required=False, help="No. epochs for training the autoencoder that embeds the social relationships. Required if rel_ne_tec=='ae'")
train_parser.add_argument('spat_autoencoder_epochs', type=int, default=100, required=False, help="No. epochs for training the autoencoder that embeds the social relationships. Required if spat_ne_tec=='ae'")
train_parser.add_argument('rel_adj_matrix_url', type=str, required=False, help="Link to the .csv file containing the relational adjacency matrix. Required if rel_ne_tec in ['none', 'ae', 'pca']")
train_parser.add_argument('spat_adj_matrix_url', type=str, required=False, help="Link to the .csv file containing the spatial adjacency matrix. Required if spat_ne_tec in ['none', 'ae', 'pca']")
train_parser.add_argument('id2idx_rel_url', type=str, required=False, help="Link to the .pkl file with the mapping between user IDs and the index of its row in the relational adjacency matrix. Required if rel_ne_tec in ['none', 'ae', 'pca']")
train_parser.add_argument('id2idx_spat_url', type=str, required=False, help="Link to the .pkl file with the mapping between user IDs and the index of its row in the spatial adjacency matrix. Required if spat_ne_tec in ['none', 'ae', 'pca']")


@api.route("/node_classification/train", methods=['POST'])
class Train(Resource):
    @api.expect(train_parser)
    def post(self):
        train_params = train_parser.parse_args(request)
        job_id = train_params['job_id']
        WORD_EMB_SIZE = train_params['kc']
        window = train_params['window']
        w2v_epochs = train_params['w2v_epochs']
        rel_ne_technique = train_params['rel_ne_tec']
        spat_ne_technique = train_params['spat_ne_tec']
        rel_node_embedding_size = train_params['kr']
        spat_node_embedding_size = train_params['ks']
        n_of_walks_rel = train_params['n_of_walks_rel']
        n_of_walks_spat = train_params['n_of_walks_spat']
        walk_length_rel = train_params['walk_length_rel']
        walk_length_spat = train_params['walk_length_spat']
        p_rel = train_params['p_rel']
        p_spat = train_params['p_spat']
        q_rel = train_params['q_rel']
        q_spat = train_params['q_spat']
        n2v_epochs_rel = train_params['n2v_epochs_rel']
        n2v_epochs_spat = train_params['n2v_epochs_spat']
        rel_ae_epochs = train_params['rel_autoencoder_epochs']
        spat_ae_epochs = train_params['spat_autoencoder_epochs']
        rel_adj_matrix_url = train_params['rel_adj_matrix_url']
        spat_adj_matrix_url = train_params['spat_adj_matrix_url']
        id2idx_rel_url = train_params['id2idx_rel_url']
        id2idx_spat_url = train_params['id2idx_spat_url']

        """task = train_task.apply_async(args=[content_url, WORD_EMB_SIZE, window, w2v_epochs, rel_ne_technique, spat_ne_technique, rel_node_embedding_size,
                                            spat_node_embedding_size, social_network_url, spatial_network_url, n_of_walks_rel, n_of_walks_spat, walk_length_rel,
                                            walk_length_spat, p_rel, p_spat, q_rel, q_spat, n2v_epochs_rel, n2v_epochs_spat, spat_ae_epochs, rel_ae_epochs,
                                            rel_adj_matrix_url, spat_adj_matrix_url, id2idx_rel_url, id2idx_spat_url])"""
        task = train_task.apply_async(args=[job_id, WORD_EMB_SIZE, window, w2v_epochs, rel_ne_technique, spat_ne_technique, rel_node_embedding_size,
                                            spat_node_embedding_size, n_of_walks_rel, n_of_walks_spat, walk_length_rel,
                                            walk_length_spat, p_rel, p_spat, q_rel, q_spat, n2v_epochs_rel, n2v_epochs_spat, spat_ae_epochs, rel_ae_epochs,
                                            rel_adj_matrix_url, spat_adj_matrix_url, id2idx_rel_url, id2idx_spat_url])
        return jsonify({"Job id": task.id})


predict_parser = reqparse.RequestParser()
predict_parser.add_argument("preprocess_job_id", required=True, help="ID of the preprocessing Job that created the dataset")
predict_parser.add_argument("train_job_id", required=True, help="ID of the training Job that created the model you want to use")
predict_parser.add_argument("user_ids", required=True, type=str, help="Comma-separated list of users you want to classify")

@api.route("/node_classification/predict", methods=["POST"])
class Predict(Resource):
    @api.expect(predict_parser)
    def post(self):
        predict_params = predict_parser.parse_args(request)
        train_job_id = predict_params['train_job_id']
        preprocess_job_id = predict_params['preprocess_job_id']
        splitted_ids = predict_params['user_ids'].split(",")
        user_ids = [int(el.strip()) for el in splitted_ids]
        if not exists(join(JOBS_DIR, train_job_id)) or not exists(join(JOBS_DIR, train_job_id, "techniques.txt")):
            abort(400, "ERROR: the learning job id is not valid or not existent")
        if not exists(join(JOBS_DIR, preprocess_job_id)):
            abort(400, "ERROR: the preprocessing job id is not valid or not existent")
        with open(join(JOBS_DIR, train_job_id, "techniques.txt"), 'r') as f:
            params = [l.strip() for l in f.readlines()]
        rel_technique = params[0]
        spat_technique = params[1]
        we_dim = params[2]
        task = train_task.AsyncResult(train_job_id)
        if task.state == "FAILURE":
            abort(400, "ERROR: the job id that you provided corresponds to a job that didn't successfully complete.")
        elif task.state == 'PROGRESS' or task.state == 'STARTED':
            abort(204, "The training task has not completed yet. You can use the 'task_status' endpoint to check for the task state")
        elif task.state == 'SUCCESS' or (task.state == 'PENDING' and exists(join(JOBS_DIR, train_job_id))):
            pred = classify_users(preprocess_job_id=join(JOBS_DIR, preprocess_job_id), train_job_id=join(JOBS_DIR, train_job_id),
                                  user_ids=user_ids, content_filename=CONTENT_FILENAME, id2idx_rel_fname=ID2IDX_REL_FILENAME,
                                  id2idx_spat_fname=ID2IDX_SPAT_FILENAME, rel_adj_mat_fname=REL_ADJ_MAT_FILENAME, spat_adj_mat_fname=SPAT_ADJ_MAT_FILENAME,
                                  rel_technique=rel_technique, spat_technique=spat_technique, we_dim=we_dim)
            readable_preds = {}
            for k in pred.keys():
                if pred[k] == 0:
                    readable_preds[k] = "safe"
                elif pred[k] == 1:
                    readable_preds[k] = "risky"
                else:
                    readable_preds[k] = pred[k]
            return jsonify(readable_preds)

@api.route("/node_classification/task_status/<task_id>")
class TaskStatus(Resource):
    def get(self, task_id):
        task = train_task.AsyncResult(task_id)
        response = {
            'state': task.state,
        }
        if task.info:
            response['info'] = task.info
        return jsonify(response)


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)
