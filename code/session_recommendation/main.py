from flask import Flask, jsonify, request
from session_recommendation.hgru4rec.hgru4rec import model_fn as hgru4rec_model_fn
import json
import tensorflow as tf
from dg_ml_core.datastores import gcs_utils
from dg_ml_core.collections import dict_ops
import argparse


app = Flask('SessionBasedRecommendations')
READY = False
MODEL = None
EMBEDDING_DICT = dict()


@app.route('/Readiness/', methods=['GET'])
def readiness():
    if READY:
        return jsonify({"message": "API is ready"}), 200
    return jsonify({"message": "API is not ready"}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Session Recommendation App is live"}), 200


def initialize_app(model_name, model_path, embedding_dict_path, params_path):
    global READY
    global MODEL
    global EMBEDDING_DICT

    EMBEDDING_DICT = dict_ops.load_dict(embedding_dict_path)
    params = dict_ops.load_dict(params_path)

    gcs_utils.download_folder_to_target(model_path, '/model_data')

    if model_name == 'hgru4rec':
        MODEL = tf.estimator.Estimator(
            model_fn=hgru4rec_model_fn,
            model_dir='/model_data',
            params=params)
    else:
        raise RuntimeError(f'Model {model_name} unknown')
    READY = True


@app.route('/predict/', methods=['POST'])
def predict():
    if request.headers["Content-Type"] == 'application/json':
        request_data = request.get_json()
        user_id = request_data['userId']
        product_id = request_data['productId']
        session_start = request_data['sessionStart']

        if str(user_id) not in EMBEDDING_DICT['User']['ToEmbedding']:
            return (jsonify({
                "error": {
                    "code": 404,
                    "message": "User is not supported"
                }}), 404)

        if str(product_id) not in EMBEDDING_DICT['Product']['ToEmbedding']:
            return (jsonify({
                "error": {
                    "code": 404,
                    "message": "Product is not supported"
                }}), 404)
            
        else:
            features = dict()
            features['UserId'] = [user_id]
            features['UserEmbeddingId'] = [EMBEDDING_DICT['User']['ToEmbedding'][str(user_id)]]
            features['ProductId'] = product_id
            features['EmbeddingId'] = [EMBEDDING_DICT['Product']['ToEmbedding'][str(product_id)]]
            features['SessionChanged'] = int(session_start)
            features['UserChanged'] = 0
            features['Epoch'] = 0
            features['LastSessionEvent'] = 1

            predictions = MODEL.predict(features)

            product_ids = map(lambda x: EMBEDDING_DICT['Prodct']['FromEmbedding'][str(x)], predictions)

            return (jsonify({
                'predictions': list(product_ids)
            }), 200)

    else:
        return (jsonify({
            "error": {
                "code": 500,
                "message": "Invalid input format"
            }}), 400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--embedding_dict_path', type=str)
    parser.add_argument('--params_path', type=str)

    args = parser.parse_args()

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

    initialize_app(**args)
