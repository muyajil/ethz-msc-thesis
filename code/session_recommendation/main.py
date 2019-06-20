from flask import Flask, jsonify, request
from session_recommendation.hgru4rec.hgru4rec import model_fn as hgru4rec_model_fn
import json
import tensorflow as tf
from dg_ml_core.datastores import gcs_utils
from dg_ml_core.collections import dict_ops
import argparse
import logging
import os
import numpy as np


app = Flask('SessionBasedRecommendations')
READY = False
MODEL = None
EMBEDDING_DICT = dict()


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


@app.route('/Readiness/', methods=['GET'])
def readiness():
    if READY:
        return jsonify({"message": "API is ready"}), 200
    return jsonify({"message": "API is not ready"}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Session Recommendation App is live"}), 200


def initialize_app(model_name, model_path, embedding_dict_path):
    global READY
    global MODEL
    global EMBEDDING_DICT

    app.logger.info('Loading embedding dict...')
    EMBEDDING_DICT = dict_ops.load_dict(embedding_dict_path)
    app.logger.info('Loading model params...')
    params = dict_ops.load_dict(model_path + 'params.json')

    app.logger.info('Downloading model artifacts...')
    gcs_utils.download_folder_to_target(model_path, '/model_data')

    if model_name == 'hgru4rec':
        MODEL = tf.estimator.Estimator(
            model_fn=hgru4rec_model_fn,
            model_dir='/model_data',
            params=params)
    else:
        raise RuntimeError(f'Model {model_name} unknown')
    
    app.logger.info('Finished initialization!')
    READY = True


@app.route('/predict/', methods=['POST'])
def predict():
    app.logger.info('Started prediction')
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
            features['UserId'] = np.array([user_id])
            features['UserEmbeddingId'] = np.array([EMBEDDING_DICT['User']['ToEmbedding'][str(user_id)]])
            features['ProductId'] = np.array([product_id])
            features['EmbeddingId'] = np.array([EMBEDDING_DICT['Product']['ToEmbedding'][str(product_id)]])
            features['SessionChanged'] = np.array([int(session_start)])
            features['UserChanged'] = np.array([0])
            features['Epoch'] = np.array([0])
            features['LastSessionEvent'] = np.array([0])

            predictions = MODEL.predict(lambda: features)

            product_ids = map(lambda x: EMBEDDING_DICT['Product']['FromEmbedding'][str(x)], list(predictions))

            app.logger.info('Recommended {} to {}'.format(','.join(product_ids), user_id))

            return (jsonify({
                'predictions': list(product_ids)
            }), 200)

    else:
        return (jsonify({
            "error": {
                "code": 500,
                "message": "Invalid input format"
            }}), 400)

initialize_app(os.environ['MODEL_NAME'], os.environ['MODEL_PATH'], os.environ['EMBEDDING_DICT_PATH'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--embedding_dict_path', type=str)

    args = parser.parse_args()

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

    initialize_app(**args)
