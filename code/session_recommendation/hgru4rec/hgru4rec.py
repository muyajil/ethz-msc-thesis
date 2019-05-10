import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.contrib.cudnn_rnn import CudnnGRU
from tensorflow.metrics import precision_at_k, recall_at_k


def mrr_metric(labels, predictions, k=None, weights=None,
               metrics_collections=None,
               updates_collections=None,
               name=None):

    with tf.name_scope(name, 'mrr_metric', [predictions, labels, weights]) as scope:

        if k is None:
            k = predictions.get_shape().as_list()[-1]
        _, pred_embedding_ids = tf.nn.top_k(predictions, k)
        labels = tf.broadcast_to(
            tf.expand_dims(labels, 1),
            tf.shape(pred_embedding_ids))

        ranked_indices = tf.where(
            tf.equal(tf.cast(pred_embedding_ids, tf.int64), labels))[:, 1]

        inverse_rank = 1/(ranked_indices + 1)
        m_rr, update_mrr_op = tf.metrics.mean(
            inverse_rank,
            weights=weights,
            name=name)

        if metrics_collections:
            tf.add_to_collection(metrics_collections, m_rr)

        if updates_collections:
            tf.add_to_collections(updates_collections, update_mrr_op)

        return m_rr, update_mrr_op


def top1_loss_mod(logits):
    logits = tf.tanh(logits)
    logits = tf.transpose(logits)
    total_loss = tf.reduce_mean(tf.sigmoid(
        logits-tf.diag_part(logits))+tf.sigmoid(logits**2), axis=0)
    answer_loss = tf.sigmoid(tf.diag_part(logits)**2) / \
        tf.cast(tf.shape(logits)[0], tf.float32)
    loss = tf.reduce_mean(total_loss-answer_loss)
    return loss


def top1_loss(logits):

    yhat = tf.nn.softmax(logits)

    yhatT = tf.transpose(yhat)

    term1 = tf.reduce_mean(
        tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) +
        tf.nn.sigmoid(yhatT**2), axis=0)

    term2 = tf.divide(
        tf.nn.sigmoid(tf.diag_part(yhat)**2),
        tf.cast(tf.shape(logits)[0], tf.float32))
    loss = tf.reduce_mean(term1 - term2)
    return loss


def model_fn(features, labels, mode, params):

    batch_size = features['UserId'].shape[0]

    num_ended_sessions = tf.get_variable(
        'num_ended_sessions',
        shape=(),
        initializer=tf.zeros_initializer(),
        trainable=False,
        dtype=tf.int64)

    num_ended_users = tf.get_variable(
        'num_ended_users',
        shape=(),
        initializer=tf.zeros_initializer(),
        trainable=False,
        dtype=tf.int64)

    num_rel_sessions = tf.get_variable(
        'num_rel_sessions',
        shape=(),
        initializer=tf.zeros_initializer(),
        trainable=False,
        dtype=tf.int64)

    # Update stats
    inc_op_session = num_ended_sessions.assign_add(
        tf.reduce_sum(features['SessionChanged'])
    )

    inc_op_user = num_ended_users.assign_add(
        tf.reduce_sum(features['UserChanged'])
    )

    inc_op_rel_session = num_rel_sessions.assign_add(
        tf.subtract(
            tf.cast(batch_size, tf.int64),
            tf.reduce_sum(features['LastSessionEvent'])))

    tf.summary.scalar('observe/num_ended_sessions', num_ended_sessions)

    tf.summary.scalar('observe/num_ended_users', num_ended_users)

    tf.summary.scalar(
        'observe/avg_relevant_sessions',
        num_rel_sessions/tf.train.get_or_create_global_step())

    tf.summary.scalar('observe/epochs', features['Epoch'][0])

    # Hidden states of session_rnn
    session_hidden_states_var = tf.get_variable(
        'session_hidden_states',
        initializer=tf.initializers.random_normal(),
        shape=(batch_size, params['session_rnn_units']),
        trainable=False)

    tf.summary.histogram(
            'variables/session_hidden_states',
            session_hidden_states_var)

    # Softmax weights to map RNN output to product space
    softmax_weights = tf.get_variable(
        'softmax_weights',
        shape=(params['num_products'], params['session_rnn_units']))

    # Biases for above
    softmax_biases = tf.get_variable(
        'softmax_biases',
        shape=(params['num_products'],))

    session_rnn = GRU(
        # params['session_rnn_layers'],
        params['session_rnn_units'],
        return_state=True,
        implementation=2,
        dropout=params['session_dropout'],
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        recurrent_initializer=tf.contrib.layers.xavier_initializer(),
        name='session_rnn')

    # Get session hidden states to update
    indices_to_update = tf.squeeze(
        tf.where(
            tf.cast(features['SessionChanged'], tf.bool),
            name='get_indices_to_update'
        ),
        axis=1
    )

    if params['use_user_rnn']:
        tf.logging.info('Adding user_rnn')
        user_rnn = GRU(
            # params['user_rnn_layers'],
            params['user_rnn_units'],
            return_state=True,
            implementation=2,
            dropout=params['user_dropout'],
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            recurrent_initializer=tf.contrib.layers.xavier_initializer(),
            name='user_rnn')

        # User Embedding, updated by user_rnn
        user_embeddings_var = tf.get_variable(
            'user_embeddings',
            initializer=tf.initializers.random_normal(),
            shape=(params['num_users'], params['user_rnn_units']),
            trainable=False)

        tf.summary.histogram(
            'variables/user_embeddings',
            user_embeddings_var)

        # Layer to predict new session initialization
        user2session_layer = Dense(
            params['session_rnn_units'],
            input_shape=(params['user_rnn_units'],),
            activation='tanh',
            name='user2session_layer')

        # Dropout layer for session initialization
        user2session_dropout = Dropout(params['init_dropout'])

        session_states_to_update = tf.gather(
            session_hidden_states_var,
            indices_to_update,
            name='get_session_states_to_update'
        )

        # Get user embeddings to update
        user_embedding_ids_to_update = tf.gather(
            features['UserEmbeddingId'],
            indices_to_update
        )

        user_embeddings_to_update = tf.nn.embedding_lookup(
            user_embeddings_var,
            user_embedding_ids_to_update
        )

        # Compute new user representation for all users in current batch
        new_session_hidden_states_seed, new_user_embeddings = user_rnn.apply(
            tf.expand_dims(session_states_to_update, 1),
            initial_state=user_embeddings_to_update)

        # Predict new session initialization for next session
        new_session_hidden_states = user2session_layer.apply(
            new_session_hidden_states_seed)

        new_session_hidden_states = user2session_dropout.apply(
            new_session_hidden_states)

        # Update user embeddings
        scattered_new_embeddings = tf.scatter_nd(
            tf.cast(tf.expand_dims(
                user_embedding_ids_to_update, axis=1), tf.int32),
            new_user_embeddings,
            tf.shape(user_embeddings_var)
        )

        scattered_mask = tf.scatter_nd(
            tf.cast(tf.expand_dims(
                user_embedding_ids_to_update, axis=1), tf.int32),
            tf.ones(tf.shape(user_embedding_ids_to_update)),
            (params['num_users'],)
        )

        merged_user_embeddings = tf.where(
            tf.cast(scattered_mask, tf.bool),
            scattered_new_embeddings,
            user_embeddings_var)

        usr_assign_op = tf.assign(
            user_embeddings_var,
            merged_user_embeddings,
            name='assign_user_embeddings'
        )

        with tf.control_dependencies([usr_assign_op]):

            # Update session hidden states
            scattered_new_states = tf.scatter_nd(
                tf.cast(tf.expand_dims(indices_to_update, axis=1), tf.int32),
                new_session_hidden_states,
                tf.shape(session_hidden_states_var)
            )

            merged_session_hidden_states = tf.where(
                tf.cast(features['SessionChanged'], tf.bool),
                scattered_new_states,
                session_hidden_states_var)

    else:
        merged_session_hidden_states = tf.where(
            tf.cast(features['SessionChanged'], tf.bool),
            tf.random.normal(tf.shape(session_hidden_states_var)),
            session_hidden_states_var)

    sess_assign_op = tf.assign(
            session_hidden_states_var,
            merged_session_hidden_states,
            name='assign_session_hidden_states'
        )

    with tf.control_dependencies([sess_assign_op]):
        session_hidden_states = tf.identity(
            merged_session_hidden_states,
            name='update_session_hidden_states')

    with tf.control_dependencies([
            inc_op_session,
            inc_op_user,
            inc_op_rel_session]):
        # Compute mask for ending sessions
        relevant_indices = tf.squeeze(
            tf.where(
                tf.logical_not(
                    tf.cast(features['LastSessionEvent'], tf.bool))),
            axis=1)

    # Get session hidden states
    relevant_session_hidden_states = tf.gather(
            session_hidden_states,
            relevant_indices
        )

    # Get relevant One Hot Encodings of Products
    relevant_product_ids = tf.gather(
        features['EmbeddingId'],
        relevant_indices
    )

    relevant_one_hots = tf.one_hot(
        relevant_product_ids,
        params['num_products']
    )

    # Apply Session RNN -> get new hidden states and predictions
    predictions, new_session_hidden_states = session_rnn.apply(
        tf.expand_dims(relevant_one_hots, 1),
        initial_state=relevant_session_hidden_states)

    # Update session hidden states for relevant sessions
    scattered_new_states = tf.scatter_nd(
        tf.cast(tf.expand_dims(relevant_indices, axis=1), tf.int32),
        new_session_hidden_states,
        tf.shape(session_hidden_states)
    )

    scattered_mask = tf.scatter_nd(
        tf.cast(tf.expand_dims(relevant_indices, axis=1), tf.int32),
        tf.ones(tf.shape(relevant_indices)),
        (batch_size,)
    )

    merged_session_hidden_states = tf.where(
        tf.cast(scattered_mask, tf.bool),
        scattered_new_states,
        session_hidden_states
    )

    sess_assign_op_sess = tf.assign(
            session_hidden_states_var,
            merged_session_hidden_states,
            name='assign_session_hidden_states_sess'
        )

    with tf.control_dependencies([sess_assign_op_sess]):

        # Extract relevant labels
        relevant_labels = tf.gather(
            labels['EmbeddingId'],
            relevant_indices
        )

    # Get softmax weights for relevant labels
    samples_softmax_weights = tf.nn.embedding_lookup(
        softmax_weights,
        relevant_labels,
        name='get_samples_softmax_weights')

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
    top_1_loss = top1_loss_mod(loss_relevant_logits)

    # Compute logits for product predictions
    logits = tf.matmul(
        predictions,
        softmax_weights,
        transpose_b=True) + softmax_biases

    _, ranked_predictions = tf.nn.top_k(logits, params['num_predictions'])
    top_predictions = ranked_predictions[:, 0]

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=relevant_labels))

    precision_at_10 = precision_at_k(
        labels=relevant_labels,
        predictions=logits,
        k=10,
        name='compute_precision_at_k')

    recall_at_10 = recall_at_k(
        labels=relevant_labels,
        predictions=logits,
        k=10,
        name='compute_recall_at_10')

    mrr = mrr_metric(
        labels=relevant_labels,
        predictions=logits,
        name='compute_mrr')

    tf.summary.histogram('observe/labels', relevant_labels)
    tf.summary.histogram('observe/logits', logits)
    tf.summary.histogram('observe/top_predictions', top_predictions)

    if params['loss_function'] == 'top_1':
        loss = top_1_loss
    elif params['loss_function'] == 'cross_entropy':
        loss = cross_entropy_loss

    if mode == tf.estimator.ModeKeys.TRAIN:

        tf.summary.scalar(
            'train_metrics/top_1_loss',
            top_1_loss)

        tf.summary.scalar(
            'train_metrics/cross_entropy_loss',
            cross_entropy_loss)

        tf.summary.scalar(
            'train_metrics/precision_at_10',
            precision_at_10[1])

        tf.summary.scalar(
            'train_metrics/recall_at_10',
            recall_at_10[1])

        tf.summary.scalar(
            'train_metrics/mrr',
            mrr[1]
        )

        if params['optimizer'] == 'adagrad':

            optimizer = tf.train.AdagradOptimizer(
                learning_rate=params['learning_rate'])

        elif params['optimizer'] == 'momentum':

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=params['learning_rate'],
                momentum=params['momentum'])

        elif params['optimizer'] == 'adam':

            optimizer = tf.train.AdamOptimizer(
                learning_rate=params['learning_rate'])

        elif params['optimizer'] == 'sgd':

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=params['learning_rate'])

        grads_and_vars = optimizer.compute_gradients(loss)

        grads_and_vars = [(
            tf.clip_by_norm(
                grad,
                params['clip_gradients_at']),
            var) for grad, var in grads_and_vars]

        for grad, var in grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tf.summary.histogram(
                        "gradients/{}".format(var.name.replace(':', '_')),
                        grad.values)
                else:
                    tf.summary.histogram(
                        "gradients/{}".format(var.name.replace(':', '_')),
                        grad)

            if isinstance(var, tf.IndexedSlices):
                tf.summary.histogram(
                    "variables/{}".format(var.name.replace(':', '_')),
                    var.values)
            else:
                tf.summary.histogram(
                    "variables/{}".format(var.name.replace(':', '_')),
                    var)

        train_op = optimizer.apply_gradients(
            grads_and_vars,
            global_step=tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            'eval_metrics/precision_at_10': precision_at_10,
            'eval_metrics/recall_at_10': recall_at_10,
            'eval_metrics/mrr': mrr
        }

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=ranked_predictions)