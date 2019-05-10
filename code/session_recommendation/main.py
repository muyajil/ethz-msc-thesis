from flask import Flask, jsonify, request
from hgru4rec.hgru4rec import model_fn as hgru4rec_model_fn
import json

app = Flask('SessionBasedRecommendations')
READY = False
MODEL = None
EMBEDDING_DICT = None


@app.route('/Readiness/', methods=['GET'])
def readiness():
    if READY:
        return jsonify({"message": "API is ready"}), 200
    return jsonify({"message": "API is not ready"}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Session Recommendation App is live"}), 200


def initialize_app(model_name, model_path, embedding_dict_path, params):
    global READY
    global MODEL
    global EMBEDDING_DICT
    # TODO: does gcs work or is it necessary to have the model in the image?
    # TODO: params loaded from local/remote json -> maybe save them after training in the model dir?
    # TODO: Load embedding dict
    if model_name == 'hgru4rec':
        MODEL = tf.estimator.Estimator(
            model_fn=hgru4rec_model_fn,
            model_dir=model_path,
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

        if (str(user_id) not in EMBEDDING_DICT['User']['ToEmbedding'] or
                str(product_id) not in EMBEDDING_DICT['Product']['ToEmbedding']):
            return (jsonify({
                "error": {
                    "code": 404,
                    "message": "User or product is not supported"
                }}), 404)

        else:
            # TODO: build input features (user changed always if session starts)
            # TODO: get embedding ids
            # TODO: predict result
            pass

    else:
        return (jsonify({
            "error": {
                "code": 500,
                "message": "Invalid input format"
            }}), 400)


initialize_app()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
