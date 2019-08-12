from flask import Flask, jsonify, request
import json
from dg_ml_core.datastores import gcs_utils
from dg_ml_core.collections import dict_ops
import argparse
import logging
import os
import numpy as np
import requests


app = Flask('SessionBasedRecommendations')
READY = False
EMBEDDING_DICT = dict()
USER_EMBEDDINGS = dict()
PRODUCT_EMBEDDINGS = dict()
SESSION_EMBEDDINGS = dict()
MODEL_NAME = ''


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
    MODEL_NAME = os.environ['MODEL_NAME']
    USER_EMBEDDINGS = json.load(open('/session_recommendation/models/{}/1/user_embeddings.json'.format(MODEL_NAME)))
    SESSION_EMBEDDINGS = json.load(open('/session_recommendation/models/{}/1/session_embeddings.json'.format(MODEL_NAME)))
    if 'embedding' in MODEL_NAME:
        PRODUCT_EMBEDDINGS = json.load(open('/session_recommendation/product_embeddings.json'))
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
                request_data['inputs']['ProductEmbeddings'] = PRODUCT_EMBEDDINGS[str(product_id)]
            request_data['inputs']['UserEmbeddings'] = USER_EMBEDDINGS[str(user_id)]
            request_data['inputs']['SessionEmbeddings'] = SESSION_EMBEDDINGS[str(session_id)]

            response = requests.post(
                'http://localhost:8501/v1/models/session_recommendation:predict',
                data=json.dumps(request_data)
            )

            response = json.loads(response.text)

            product_ids = map(lambda x: EMBEDDING_DICT['Product']['FromEmbedding'][str(x)], list(response['RankedPredictions']))

            SESSION_EMBEDDINGS[str(session_id)] = list(response['SessionEmbeddings'])[0]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)