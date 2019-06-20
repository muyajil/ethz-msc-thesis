import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.contrib.cudnn_rnn import CudnnGRU
from tensorflow.metrics import precision_at_k, recall_at_k
from functools import partial
from metrics import mrr_at_k, top1_loss
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)


class HGRU4RECSummaries(object):
    logits = None
    cross_entropy_loss = None
    top1_loss = None
    top_predictions = None
    mrr_at_10 = None
    precision_at_10 = None
    recall_at_10 = None
    session_embeddings = None
    user_embeddings = None


class HGRU4RECFeatures(object):
    user_embeddings = None
    session_embeddings = None
    product_embedding_ids = None
    session_changed = None
    user_ids = None


class HGRU4RecLosses(object):
    cross_entropy_loss = None
    top1_loss = None


class HGRU4RecMetrics(object):
    mrr_at_10 = None
    precision_at_10 = None
    recall_at_10 = None

    mrr_update_op = None
    precision_update_op = None
    recall_update_op = None


class HGRU4RecOps(object):
    features = HGRU4RECFeatures()
    labels = None

    metrics = HGRU4RecMetrics()
    summaries = HGRU4RECSummaries()
    global_step = None

    user_embeddings = None
    session_embeddings = None

    ranked_predictions = None

    losses = HGRU4RecLosses()
    optimizer = None
    grads_and_vars = None

    logits = None


class HGRU4Rec(object):

    def __init__(self, config):
        self.user_embeddings = dict()
        self.session_embeddings = dict()
        self._config = config
        self._ops = HGRU4RecOps()
        self.logger = logging.getLogger('HGRU4Rec')

    def _update_embeddings(self, embeddings, updates, indices):
        for idx, update in zip(indices, updates):
            embeddings[idx] = update.tolist()

    def _get_or_initialize_embeddings(self, embeddings, indices, num_units):
        result = []
        for idx in indices:
            if idx not in embeddings:
                embeddings[idx] = np.random.normal(size=(num_units)).tolist()
            result.append(embeddings[idx])

        return np.array(result)

    def _preprocess(self, batch):
        batch = batch.loc[batch['LastSessionEvent'] == 0].reset_index()

        user_embeddings = self._get_or_initialize_embeddings(
            self.user_embeddings,
            batch['UserId'],
            self._config['user_rnn_units'])

        session_embeddings = self._get_or_initialize_embeddings(
            self.session_embeddings,
            batch['SessionId'],
            self._config['session_rnn_units'])

        batch = batch.astype(dtype={'SessionChanged': np.bool})

        return batch, user_embeddings, session_embeddings

    def train(self, dataset):
        # TODO: setup model, restore if necessary
        with tf.Session() as sess:
            local_init_op = tf.local_variables_initializer()
            global_init_op = tf.global_variables_initializer()
            sess.run(local_init_op)
            sess.run(global_init_op)

            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                self._config['log_dir'], graph=sess.graph)

            for epoch in range(self._config['epochs']):
                batches = dataset.feature_and_label_generator()

                for batch in batches:
                    batch, user_embeddings, session_embeddings = self._preprocess(
                        batch)

                    fetches = [summaries,
                               self._ops.session_embeddings,
                               self._ops.global_step,
                               self._ops.losses.cross_entropy_loss,
                               self._ops.losses.top1_loss,
                               self._ops.optimizer,
                               self._ops.metrics.mrr_update_op,
                               self._ops.metrics.precision_update_op,
                               self._ops.metrics.recall_update_op]

                    if self._config['use_user_rnn']:
                        fetches.append(self._ops.user_embeddings)

                    result = sess.run(fetches,
                                      feed_dict={
                                          self._ops.features.user_embeddings: user_embeddings,
                                          self._ops.features.session_embeddings: session_embeddings,
                                          self._ops.features.product_embedding_ids: batch['EmbeddingId'],
                                          self._ops.features.user_ids: batch['UserId'],
                                          self._ops.features.session_changed: batch['SessionChanged'],
                                          self._ops.labels: batch['LabelEmbeddingId']
                                      })

                    if self._config['use_user_rnn']:
                        (summary_str,
                         updated_session_embeddings,
                         global_step,
                         cross_entropy_loss,
                         top1_loss,
                         _, _, _, _,
                         updated_user_embeddings) = result
                    else:
                        (summary_str,
                         updated_session_embeddings,
                         global_step,
                         cross_entropy_loss,
                         top1_loss,
                         _, _, _, _) = result

                    writer.add_summary(summary_str, global_step)

                    if self._config['use_user_rnn']:

                        self._update_embeddings(
                            self.user_embeddings,
                            updated_user_embeddings,
                            batch['UserId'])

                    self._update_embeddings(
                        self.session_embeddings,
                        updated_session_embeddings,
                        batch['SessionId'])

                    if global_step % 100 == 0:

                        self.logger.info(
                            f"EPOCH: {epoch} - STEP: {global_step} - CE: {cross_entropy_loss} - TOP1: {top1_loss}")

                    if global_step % self._config['eval_every_steps'] == 0:
                        self._save(global_step)
                        self.validate()

                    if self._stopping_condition_met():
                        # TODO: gracefully exit and save checkpoint
                        pass

                self._save(global_step, export_model=True)

    def validate(self):
        # TODO: Implement validation of model
        pass

    def _stopping_condition_met(self):
        # TODO: Implement stopping conditions (early stopping, max steps etc)
        pass

    def _restore(self):
        # TODO: Implement restoring mechanism
        pass

    def _save(self, global_step, export_model=False):
        # TODO: Implement checkpoint saving
        if export_model:
            # TODO: Export model as a servable
            pass
        pass

    def predict(self):
        # TODO: use sess.run([self.ops.ranked_predictions]) to get the ranked predictions
        pass

    def _user_rnn(self, user_embeddings, session_embeddings):
        with tf.variable_scope("user_rnn"):
            user_rnn = GRU(
                # self.config['user_rnn_layers'],
                self._config['user_rnn_units'],
                return_state=True,
                implementation=2,
                dropout=self._config['user_dropout'],
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                recurrent_initializer=tf.contrib.layers.xavier_initializer(),
                name='user_rnn')

            # Layer to predict new session initialization
            user2session_layer = Dense(
                self._config['session_rnn_units'],
                input_shape=(self._config['user_rnn_units'],),
                activation='tanh',
                name='user2session_layer')

            # Dropout layer for session initialization
            user2session_dropout = Dropout(
                self._config['init_dropout'],
                name='user2session_dropout')

        new_session_embeddings_seed, new_user_embeddings = user_rnn(
            tf.expand_dims(session_embeddings, 1),
            initial_state=user_embeddings)

        # Predict new session initialization for next session
        new_session_embeddings = user2session_dropout(user2session_layer(
            new_session_embeddings_seed))

        return new_session_embeddings, new_user_embeddings

    def _session_rnn(self,
                     user_embeddings,
                     session_embeddings,
                     session_changed,
                     product_ids):

        with tf.variable_scope("session_rnn"):

            session_rnn = GRU(
                # self.config['session_rnn_layers'],
                self._config['session_rnn_units'],
                return_state=True,
                implementation=2,
                dropout=self._config['session_dropout'],
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                recurrent_initializer=tf.contrib.layers.xavier_initializer(),
                name='session_rnn')

            output_layer = Dense(
                self._config['num_products'],
                input_shape=(self._config['session_rnn_units'],),
                activation='tanh',
                name='output_layer')

        if self._config['use_user_rnn']:
            new_session_embeddings, new_user_embeddings = self._user_rnn(
                user_embeddings,
                session_embeddings)
        else:
            new_session_embeddings = tf.random.normal(
                tf.shape(session_embeddings))

        session_embeddings = tf.where(
            session_changed,
            new_session_embeddings,
            session_embeddings)

        one_hot_encodings = tf.one_hot(
            product_ids,
            self._config['num_products'])

        predictions, session_embeddings = session_rnn(
            tf.expand_dims(one_hot_encodings, 1),
            initial_state=session_embeddings)

        logits = output_layer(predictions)

        if self._config['use_user_rnn']:
            return logits, session_embeddings, new_user_embeddings
        else:
            return logits, session_embeddings

    def _setup_summaries(self):
        top_predictions = self._ops.ranked_predictions[:, 0]

        (self._ops.metrics.mrr_at_10,
         self._ops.metrics.mrr_update_op) = mrr_at_k(
            labels=self._ops.labels,
            logits=self._ops.logits,
            k=10,
            name='compute_mrr')

        tf.summary.scalar(
            'metrics/mrr_at_10',
            self._ops.metrics.mrr_at_10)

        (self._ops.metrics.precision_at_10,
         self._ops.metrics.precision_update_op) = tf.metrics.precision_at_k(
            labels=self._ops.labels,
            predictions=self._ops.logits,
            k=10,
            name='compute_precision')

        tf.summary.scalar(
            'metrics/precision_at_10',
            self._ops.metrics.precision_at_10)

        (self._ops.metrics.recall_at_10,
         self._ops.metrics.recall_update_op) = tf.metrics.recall_at_k(
            labels=self._ops.labels,
            predictions=self._ops.logits,
            k=10,
            name='compute_recall')

        tf.summary.scalar(
            'metrics/recall_at_10',
            self._ops.metrics.recall_at_10)

        for grad, var in self._ops.grads_and_vars:
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

        tf.summary.scalar('losses/cross_entropy',
                          self._ops.losses.cross_entropy_loss)
        tf.summary.scalar('losses/top1', self._ops.losses.top1_loss)
        tf.summary.histogram('observe/top_predictions', top_predictions)
        tf.summary.histogram('observe/logits', self._ops.logits)
        tf.summary.histogram('observe/labels', self._ops.labels)

        if self._config['use_user_rnn']:
            tf.summary.histogram('observe/user_embeddings',
                                 self._ops.user_embeddings)
        tf.summary.histogram('observe/session_embeddings',
                             self._ops.session_embeddings)

    def setup_model(self, restore=False):

        if restore:
            self._restore()

        # TODO: Handle restoring and saving of the model
        self._ops.global_step = tf.train.get_or_create_global_step()
        self._ops.labels = tf.placeholder(tf.int64, [None])
        self._ops.features.product_embedding_ids = tf.placeholder(tf.int64, [
                                                                  None])
        self._ops.features.session_changed = tf.placeholder(tf.bool, [None])
        self._ops.features.user_ids = tf.placeholder(tf.int64, [None])

        self._ops.features.user_embeddings = tf.placeholder(
            tf.float32,
            [None, self._config['user_rnn_units']])

        self._ops.features.session_embeddings = tf.placeholder(
            tf.float32,
            [None, self._config['session_rnn_units']])

        result = self._session_rnn(
            self._ops.features.user_embeddings,
            self._ops.features.session_embeddings,
            self._ops.features.session_changed,
            self._ops.features.product_embedding_ids)

        if self._config['use_user_rnn']:
            (self._ops.logits,
             self._ops.session_embeddings,
             self._ops.user_embeddings) = result
        else:
            (self._ops.logits,
             self._ops.session_embeddings) = result

        _, self._ops.ranked_predictions = tf.nn.top_k(
            self._ops.logits,
            self._config['num_predictions'])

        self._ops.losses.cross_entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._ops.logits,
                labels=self._ops.labels))

        sampled_logits = tf.gather(
            self._ops.logits,
            self._ops.labels,
            axis=1)

        self._ops.losses.top1_loss = tf.reduce_mean(top1_loss(sampled_logits))

        if self._config['loss_function'] == 'top_1':
            loss = self._ops.losses.top1_loss
        elif self._config['loss_function'] == 'cross_entropy':
            loss = self._ops.losses.cross_entropy_loss

        if self._config['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._config['learning_rate'])
        elif self._config['optimizer'] == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._config['learning_rate'])
        elif self._config['optimizer'] == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self._config['learning_rate'])

        self._ops.grads_and_vars = [(
            tf.clip_by_norm(
                grad,
                self._config['clip_gradients_at']),
            var) for grad, var in optimizer.compute_gradients(loss)]

        self._ops.optimizer = optimizer.apply_gradients(
            self._ops.grads_and_vars,
            global_step=tf.train.get_or_create_global_step())

        self._setup_summaries()
