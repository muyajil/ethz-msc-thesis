from flask import Flask, jsonify, request
import json
import argparse
import logging
import os
import numpy as np
import requests
from dg_ml_core.collections import dict_ops


app = Flask('SessionBasedRecommendations')
READY = False
EMBEDDING_DICT = dict()
USER_EMBEDDINGS = dict()
PRODUCT_EMBEDDINGS = dict()
SESSION_EMBEDDINGS = dict()
MODEL_NAME = ''
MODEL_ARTIFACST_BASE = 'gs://ma-muy/04_model_artifacts/'


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


def initialize_app():
    global USER_EMBEDDINGS
    global SESSION_EMBEDDINGS
    global PRODUCT_EMBEDDINGS
    global READY
    global MODEL_NAME
    global EMBEDDING_DICT
    MODEL_NAME = os.environ['MODEL_NAME']

    app.logger.info('Loading User Embeddings')
    USER_EMBEDDINGS = dict_ops.load_dict(MODEL_ARTIFACST_BASE + MODEL_NAME + '/1/user_embeddings.json')
    
    app.logger.info('Loading Session Embeddings')
    SESSION_EMBEDDINGS = dict_ops.load_dict(MODEL_ARTIFACST_BASE + MODEL_NAME + '/1/session_embeddings.json')

    if 'embedding' in MODEL_NAME:
        app.logger.info('Loading Product Embeddings')
        PRODUCT_EMBEDDINGS = dict_ops.load_dict('gs://ma-muy/product_embeddings.json')

    app.logger.info('Loading Embedding Dict')
    EMBEDDING_DICT = dict_ops.load_dict(f'gs://ma-muy/03_datasets/{os.environ["DATASET"]}/05_embedding_dict.json')

    app.logger.info('Finished Initialization')
    READY = True

@app.route('/Readiness/', methods=['GET'])
def readiness():
    if READY:
        return jsonify({"message": "API is ready"}), 200
    return jsonify({"message": "API is not ready"}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Session Recommendation App is live"}), 200


@app.route('/predict/', methods=['POST'])
def predict():
    app.logger.info('Started prediction')
    if request.headers["Content-Type"] == 'application/json':
        request_data = request.get_json()
        user_id = request_data['userId']
        product_id = request_data['productId']
        session_start = request_data['sessionStart']
        session_id = request_data['sessionId']

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
            request_data = dict()
            request_data['inputs'] = dict()
            request_data['inputs']['EmbeddingId'] = [EMBEDDING_DICT['Product']['ToEmbedding'][str(product_id)]]
            request_data['inputs']['SessionChanged'] = [session_start]
            if 'embedding' in MODEL_NAME:
                request_data['inputs']['ProductEmbeddings'] = [PRODUCT_EMBEDDINGS[str(product_id)]]
            request_data['inputs']['UserEmbeddings'] = [USER_EMBEDDINGS[str(user_id)]]
            if str(session_id) in SESSION_EMBEDDINGS:
                request_data['inputs']['SessionEmbeddings'] = [SESSION_EMBEDDINGS[str(session_id)]]
            else:
                request_data['inputs']['SessionEmbeddings'] = [np.random.normal(size=(100)).tolist()]

            response = requests.post(
                'http://localhost:8501/v1/models/{}:predict'.format(MODEL_NAME),
                data=json.dumps(request_data)
            )

            response = json.loads(response.text)
            product_ids = list(map(lambda x: EMBEDDING_DICT['Product']['FromEmbedding'][str(x)], list(response['outputs']['RankedPredictions'][0])))

            SESSION_EMBEDDINGS[str(session_id)] = response['outputs']['SessionEmbeddings'][0]
            app.logger.info('Recommended {} to {}'.format(','.join(map(lambda x: str(x), product_ids)), user_id))

            return (jsonify({
                'predictions': product_ids
            }), 200)

    else:
        return (jsonify({
            "error": {
                "code": 500,
                "message": "Invalid input format"
            }}), 400)

initialize_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)