import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.contrib.cudnn_rnn import CudnnGRU
from tensorflow.metrics import precision_at_k, recall_at_k


def top1_loss(logits, batch_size):

    yhat = tf.nn.softmax(logits)

    yhatT = tf.transpose(yhat)

    term1 = tf.reduce_mean(
        tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) +
        tf.nn.sigmoid(yhatT**2), axis=0)

    term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / batch_size
    loss = tf.reduce_mean(term1 - term2)
    return loss


def setup_variables(batch_size, params):

    # Mask describing ended sessions, true if session ended
    ended_sessions_mask = tf.get_variable(
        'ended_sessions_mask',
        shape=(batch_size,),
        initializer=tf.zeros_initializer(),
        trainable=False,
        dtype=tf.bool)

    # Mask describing ending sessions, true if session is ending
    ending_sessions_mask = tf.get_variable(
        'ending_sessions_mask',
        shape=(batch_size,),
        initializer=tf.zeros_initializer(),
        trainable=False,
        dtype=tf.bool)

    # Mask describing ended users, true if not more user events
    ended_users_mask = tf.get_variable(
        'ended_users_mask',
        shape=(batch_size,),
        initializer=tf.zeros_initializer(),
        trainable=False,
        dtype=tf.bool)

    # Hidden states of session_rnn
    session_hidden_states = tf.get_variable(
        'session_hidden_states',
        shape=(batch_size, params['session_rnn_units']),
        initializer=tf.zeros_initializer(),
        trainable=False)

    # User Embedding, updated by user_rnn
    user_embeddings = tf.get_variable(
        'user_embeddings',
        shape=(params['num_users'], params['user_rnn_units']),
        initializer=tf.zeros_initializer(),
        trainable=False
    )

    # Softmax weights to map RNN output to product space
    softmax_weights = tf.get_variable(
        'softmax_weights',
        shape=(params['num_products'], params['session_rnn_units']))

    # Biases for above
    softmax_biases = tf.get_variable(
        'softmax_biases',
        shape=(params['num_products'],))

    return (ended_sessions_mask,
            ending_sessions_mask,
            ended_users_mask,
            session_hidden_states,
            user_embeddings,
            softmax_weights,
            softmax_biases)


def setup_model(params):

    user_rnn = GRU(
        # params['user_rnn_layers'],
        params['user_rnn_units'],
        return_state=True,
        implementation=2,
        dropout=params['user_dropout'],
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        recurrent_initializer=tf.contrib.layers.xavier_initializer(),
        name='user_rnn')

    session_rnn = GRU(
        # params['session_rnn_layers'],
        params['session_rnn_units'],
        return_state=True,
        implementation=2,
        dropout=params['session_dropout'],
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        recurrent_initializer=tf.contrib.layers.xavier_initializer(),
        name='session_rnn')

    # Layer to predict new session initialization
    user2session_layer = Dense(
        params['session_rnn_units'],
        input_shape=(params['user_rnn_units'],),
        activation='tanh',
        name='user2session_layer')

    # Dropout layer for session initialization
    user2session_dropout = Dropout(params['init_dropout'])

    return (user_rnn,
            session_rnn,
            user2session_layer,
            user2session_dropout)


def model_fn(features, labels, mode, params):

    batch_size = features['UserId'].shape[0]

    (ended_sessions_mask,
        ending_sessions_mask,
        ended_users_mask,
        session_hidden_states,
        user_embeddings,
        softmax_weights,
        softmax_biases) = setup_variables(batch_size, params)

    (user_rnn,
        session_rnn,
        user2session_layer,
        user2session_dropout) = setup_model(params)

    # Ended sessions where the user did not change
    ended_sessions_same_user_mask = tf.logical_and(
        ended_sessions_mask,
        tf.logical_not(ended_users_mask)
    )

    # Get user_hidden_states to update
    # The hidden states to update are the ones where a session ended
    # but the user has stayed the same
    # The other hidden states are 0
    user_hidden_states = tf.map_fn(
        lambda x: tf.cond(
            x[1],
            true_fn=lambda: tf.nn.embedding_lookup(user_embeddings, x[0]),
            false_fn=lambda: tf.zeros(params['user_rnn_units'])
        ),
        [
            features['UserEmbeddingId'],
            ended_sessions_same_user_mask
        ],
        dtype=tf.float32,
        name='get_user_hidden_states_to_update')

    # Compute new user representation for all users in current batch
    new_session_hidden_states_seed, new_user_hidden_states = user_rnn.apply(
        tf.expand_dims(session_hidden_states, 1),
        initial_state=user_hidden_states)

    # Predict new session initialization for next session
    new_session_hidden_states = user2session_layer.apply(
        new_session_hidden_states_seed)

    new_session_hidden_states = user2session_dropout.apply(
        new_session_hidden_states
    )

    # Select new session initialization for new sessions
    session_hidden_states = tf.where(
        ended_sessions_same_user_mask,
        new_session_hidden_states,
        session_hidden_states,
        name='initialize_new_sessions')

    # Update user hidden states where the session ended
    user_embeddings = tf.scatter_update(
        user_embeddings,
        tf.boolean_mask(
            features['UserEmbeddingId'],
            ended_sessions_same_user_mask),
        tf.boolean_mask(
            new_user_hidden_states,
            ended_sessions_same_user_mask),
        name='update_user_embeddings'
    )

    tf.summary.scalar(
        'observe/num_ended_sessions',
        tf.reduce_sum(tf.cast(ended_sessions_mask, tf.int32)))

    tf.summary.scalar(
        'observe/num_ended_users',
        tf.reduce_sum(tf.cast(ended_users_mask, tf.int32)))

    tf.summary.scalar(
        'observe/num_user_updates',
        tf.reduce_sum(tf.cast(ended_sessions_same_user_mask, tf.int32)))

    # Reset Session Hidden to 0 when a user has ended in the batch before
    session_hidden_states = tf.where(
        ended_users_mask,
        tf.zeros(tf.shape(session_hidden_states)),
        session_hidden_states,
        name='reset_session_hidden_states')

    # Compute new mask for ended sessions
    ended_sessions_mask = tf.cast(
        tf.where(
            tf.equal(features['ProductId'], -1),
            tf.ones(tf.shape(ended_sessions_mask)),
            tf.zeros(tf.shape(ended_sessions_mask)),
            name='compute_ended_sessions'),
        tf.bool)

    # Compute new mask for ending sessions
    ending_sessions_mask = tf.cast(
        tf.where(
            tf.equal(labels['ProductId'], -1),
            tf.ones(tf.shape(ending_sessions_mask)),
            tf.zeros(tf.shape(ending_sessions_mask)),
            name='compute_ending_sessions'),
        tf.bool)

    # Compute new mask for ended users
    ended_users_mask = tf.cast(
        tf.where(
            tf.equal(features['UserId'], -1),
            tf.ones(tf.shape(ended_users_mask)),
            tf.zeros(tf.shape(ended_users_mask)),
            name='compute_ended_users'),
        tf.bool)

    # Relevant sessions have not ended and do not end in the next step
    relevant_sessions_mask = tf.logical_not(
        tf.logical_or(
            ended_sessions_mask,
            ending_sessions_mask))

    # Get one-hot encoding of products
    relevant_one_hots = tf.map_fn(
        lambda x: tf.cond(
            x[1],
            true_fn=lambda: tf.one_hot(x[0], params['num_products']),
            false_fn=lambda: tf.zeros(params['num_products'])
        ),
        [
            features['EmbeddingId'],
            relevant_sessions_mask
        ],
        dtype=tf.float32,
        name='get_relevant_one_hots')

    # Get session hidden states for relevant sessions
    relevant_hidden_states = tf.where(
        relevant_sessions_mask,
        session_hidden_states,
        tf.zeros(tf.shape(session_hidden_states)),
        name='get_relevant_session_hidden_states'
    )

    # Apply Session RNN -> get new hidden states and predictions
    predictions, new_session_hidden_states = session_rnn.apply(
        tf.expand_dims(relevant_one_hots, 1),
        initial_state=relevant_hidden_states)

    # Filter out irrelevant predictions
    predictions = tf.boolean_mask(
        predictions,
        relevant_sessions_mask,
        name='filter_irrelevant_predictions')

    # Update session hidden states for relevant sessions
    session_hidden_states = tf.where(
        relevant_sessions_mask,
        new_session_hidden_states,
        session_hidden_states,
        name='update_relevant_session_hidden_states')

    # Extract relevant labels
    relevant_labels = tf.boolean_mask(
        labels['EmbeddingId'],
        relevant_sessions_mask,
        name='filter_irrelevant_labels')

    # Get softmax weights for relevant labels
    samples_softmax_weights = tf.nn.embedding_lookup(
        softmax_weights,
        relevant_labels,
        name='get_sampled_softmax_weights')

    # Get softmax biases for relevant labels
    samples_softmax_biases = tf.nn.embedding_lookup(
        softmax_biases,
        relevant_labels,
        name='get_samples_softmax_biases')

    # Compute logits for computing the loss of the minibatch
    loss_relevant_logits = tf.add(
        tf.matmul(
            predictions,
            samples_softmax_weights,
            transpose_b=True,
            name='compute_softmax_logits'),
        samples_softmax_biases,
        name='add_softmax_biases')

    # Compute TOP1 loss
    loss = top1_loss(loss_relevant_logits, batch_size.value)

    # Compute logits for product predictions
    logits = tf.matmul(
        predictions,
        softmax_weights,
        transpose_b=True) + softmax_biases

    tf.summary.histogram('observe/predictions', logits)
    tf.summary.scalar(
        'observe//relevant_session',
        tf.reduce_sum(tf.cast(relevant_sessions_mask, tf.int32)))

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdagradOptimizer(
            learning_rate=params['learning_rate'])

        grads_and_vars = optimizer.compute_gradients(loss)

        for grad, var in grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tf.summary.histogram(
                        "gradients/{}".format(var.name),
                        grad.values)
                else:
                    tf.summary.histogram(
                        "gradients/{}".format(var.name),
                        grad)

            if isinstance(var, tf.IndexedSlices):
                tf.summary.histogram(
                    "variables/{}".format(var.name),
                    var.values)
            else:
                tf.summary.histogram(
                    "variables/{}".format(var.name),
                    var
                )

        train_op = optimizer.apply_gradients(
            grads_and_vars,
            global_step=tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

        precision_at_10 = precision_at_k(
            labels=relevant_labels,
            predictions=logits,
            k=10,
            name='compute_precision_at_k'
        )

        recall_at_10 = recall_at_k(
            labels=relevant_labels,
            predictions=logits,
            k=10,
            name='compute_recall_at_10'
        )

        eval_metric_ops = {
            'eval_metrics/precision_at_10': precision_at_10,
            'eval_metrics/recall_at_10': recall_at_10,
        }

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.nn.top_k(
            softmax_predictions,
            params['num_predictions'])

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
