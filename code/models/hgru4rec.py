import tensorflow as tf
from tensorflow import keras
from keras import backend as K


def model_fn(features, labels, mode, params):

    item_embeddings = keras.layers.Embedding(
        params['num_products'],
        params['embedding_dim']
    )
    
    rnn_cell = keras.layers.GRUCell()
    session_rnn = keras.layers.RNN()