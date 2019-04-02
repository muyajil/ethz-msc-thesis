import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnGRU


def model_fn(features, labels, mode, params):

    item_embeddings = tf.feature_column.categorical_column_with_hash_bucket(
        key='ProductId',
        hash_bucket_size=params['num_products'])
    
    session_rnn = CudnnGRU(
        num_layers=params['num_layers_session'],
        num_units=params['num_units_session'],
        name='session_rnn')

    user_rnn = CudnnGRU(
        num_layers=params['num_layers_user'],
        num_units=params['num_units_user'],
        name='user_rnn')