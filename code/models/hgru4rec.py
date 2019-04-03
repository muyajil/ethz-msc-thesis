import tensorflow as tf
from tensorflow.keras.layers import CuDNNGRU, RNN


def model_fn(self, features, labels, mode, params):

    batch_size = features['UserId'].shape[0]

    ended_sessions_mask = tf.get_variable(
        'ended_sessions_mask',
        shape=(batch_size,),
        initializer=tf.zeros_initializer(),
        trainable=False)

    ended_users_mask = tf.get_variable(
        'ended_users_mask',
        shape=(batch_size,),
        initializer=tf.zeros_initializer(),
        trainable=False)

    session_repr = tf.get_variable(
        'session_repr',
        shape=(batch_size, params['num_units_session']),
        initializer=tf.zeros_initializer())

    user_repr = tf.get_variable(
        'user_repr',
        shape=(batch_size, params['num_units_user']),
        initializer=tf.zeros_initializer())

    hashed_items = tf.feature_column.categorical_column_with_hash_bucket(
        key='ProductId',
        hash_bucket_size=params['num_products']+1)

    item_embeddings = tf.feature_column.embedding_column(
        hashed_items,
        dimension=params['embedding_size']
    )

    input_layer = tf.feature_column.input_layer(
        features, feature_columns=[item_embeddings])

    user_rnn_layers = [CuDNNGRU(units, return_state=True, name='user_rnn')
                       for units in params['user_units_per_layer']]

    session_rnn_layers = [
        CuDNNGRU(units, return_state=True, name='session_rnn')
        for units in params['session_units_per_layer']]

    if tf.equal(tf.reduce_any(ended_sessions_mask), tf.constant(True)):
        # TODO: for the 'True' elements of this mask we know that in the previous iteration the session has ended
        # TODO: Therefore we need to update the session initialization using the user_rnn
        pass

    if tf.equal(tf.reduce_any(ended_users_mask), tf.constant(True)):
        session_repr = tf.where(
            ended_users_mask,
            tf.zeros(tf.shape(session_repr)),
            session_repr)

        user_repr = tf.where(
            ended_sessions_mask,
            tf.zeros(tf.shape(user_repr)),
            user_repr)

    ended_sessions_mask = tf.where(features['ProductId'] == -1)
    ended_users_mask = tf.where(features['SessionId'] == -1)

    optimizer = tf.train.AdamOptimizer()

    # TODO: First, for all the users that have a new session, we need to "predict" the next session representation using the user_rnn
    # TODO: When we have the hidden states of the sessions for all users, we need to stitch the tensors back together
    # TODO: After that we can predict the new hidden states of the sessions
    # TODO: To compute the loss however we need to get some kind of softmax or similar from the session GRU, such that we can predict the product ID and compute the loss
    # TODO: That means we need to have a dense layer that produces the logits and then compute the softmax of those
    # TODO: After that we get the argmax for each case and then we have a prediction
    # TODO: To compare the prediction to the labels we probably need to lookup the labels in the hashed items lookup
    # TODO: Using those we can compute the loss function (hitrate@1 for example)
    # TODO: We can also compute the sparse_softmax_cross_entropy loss for training
    # TODO: If we want to compute a hitrate@5 we just take the top 5 items from the softmax layer, this is closer to the real world scenario
    # TODO: The hitrate is probably more sensible for the evaluation step instead of the training step
    # TODO: Training will be done using nce_loss most probably, but we can compute all three losses and train on one
    # TODO: After this is running we need to evaluate the model on existing datasets
    # TODO:
