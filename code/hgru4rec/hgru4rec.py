import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.contrib.cudnn_rnn import CudnnGRU
from tensorflow.metrics import precision_at_k, recall_at_k
from metrics import mrr_at_k, top1_loss
from ops import HGRU4RecOps
import logging
import os
import json
import time
logging.basicConfig(
    format='%(asctime)s | %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class HGRU4Rec(object):

    def __init__(self, config):
        self._product_embeddings = json.load(open('/code/product_embeddings.json'))
        self._user_embeddings = dict()
        self._session_embeddings = dict()
        self._config = config
        self._ops = HGRU4RecOps()
        self._saver = None
        self._sess = None
        self._eval_metric_history = []
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
        start = time.time()

        batch = batch.loc[batch['LastSessionEvent'] == 0].reset_index()
        batch = batch.astype(
            dtype={
                'SessionChanged': bool,
                'UserId': str,
                'SessionId': str,
                'ProductId': str})

        user_embeddings = self._get_or_initialize_embeddings(
            self._user_embeddings,
            batch['UserId'],
            self._config['user_rnn_units'])

        session_embeddings = self._get_or_initialize_embeddings(
            self._session_embeddings,
            batch['SessionId'],
            self._config['session_rnn_units'])

        product_embeddings = self._get_or_initialize_embeddings(
            self._product_embeddings,
            batch['ProductId'],
            25)

        batch = batch.astype(dtype={'SessionChanged': np.bool})

        self.logger.debug('\tPreprocessing:\t\t{:.2f} secs'.format(time.time() - start))

        return batch, user_embeddings, session_embeddings, product_embeddings

    def _postprocess(self, result, batch):

        start = time.time()

        if self._config['use_user_rnn']:

            self._update_embeddings(
                self._user_embeddings,
                result['user_embeddings'],
                batch['UserId'])

        self._update_embeddings(
            self._session_embeddings,
            result['session_embeddings'],
            batch['SessionId'])

        self.logger.debug('\tPostprocessing:\t\t{:.2f} secs'.format(time.time() - start))

    def _get_fetch_dict(self, mode):

        fetch_dict = dict()

        if mode == tf.estimator.ModeKeys.PREDICT:
            fetch_dict['ranked_preds'] = self._ops.ranked_predictions
            fetch_dict['session_embeddings'] = self._ops.session_embeddings
            return fetch_dict

        else:
            fetch_dict['global_step'] = self._ops.global_step
            fetch_dict['cross_entropy_loss'] = self._ops.losses.cross_entropy_loss
            fetch_dict['top1_loss'] = self._ops.losses.top1_loss
            fetch_dict['mrr_update_op'] = self._ops.metrics.mrr_update_op
            fetch_dict['precision_update_op'] = self._ops.metrics.precision_update_op
            fetch_dict['recall_update_op'] = self._ops.metrics.recall_update_op

            if mode == tf.estimator.ModeKeys.TRAIN:
                fetch_dict['session_embeddings'] = self._ops.session_embeddings
                if self._config['use_user_rnn']:
                    fetch_dict['user_embeddings'] = self._ops.user_embeddings
                fetch_dict['optimizer'] = self._ops.optimizer
                fetch_dict['summary_str'] = tf.summary.merge(
                    self._ops.summaries.train_summaries)
            else:
                fetch_dict['summary_str'] = tf.summary.merge(
                    self._ops.summaries.eval_summaries)
            return fetch_dict

    def _get_metrics_fetch_dict(self):
        fetch_dict = dict()
        fetch_dict['mrr'] = self._ops.metrics.mrr_at_10
        fetch_dict['precision'] = self._ops.metrics.precision_at_10
        fetch_dict['recall'] = self._ops.metrics.recall_at_10
        fetch_dict['summary_str'] = tf.summary.merge(
            self._ops.summaries.metrics_summaries)

        return fetch_dict

    # TODO: Move to abstract class
    def train(self, train_dataset, validation_dataset=None, eval_metric_key=None):
        local_init_op = tf.local_variables_initializer()
        global_init_op = tf.global_variables_initializer()
        self._sess.run(global_init_op)

        early_stop = False
        mode = tf.estimator.ModeKeys.TRAIN

        writer = tf.summary.FileWriter(
            self._config['log_dir'], graph=self._sess.graph)

        fetch_dict = self._get_fetch_dict(mode)
        metrics_fetch_dict = self._get_metrics_fetch_dict()

        for epoch in range(self._config['epochs']):
            if early_stop:
                break
            batches = train_dataset.feature_and_label_generator()

            for batch in batches:
                self.logger.debug('[START] Batch')
                start = time.time()
                self._sess.run(local_init_op)
                self.logger.debug('\tLocal Init Op:\t\t{:.2f} secs'.format(time.time() - start))
                if early_stop:
                    break

                result = self._run_step(batch, fetch_dict)
                if result is None:
                    continue
                metrics = self._get_metrics(metrics_fetch_dict)

                writer_start = time.time()

                writer.add_summary(
                    result['summary_str'], result['global_step'])

                writer.add_summary(
                    metrics['summary_str'], result['global_step'])

                self.logger.debug('\tWrite summaries:\t{:.2f} secs'.format(time.time() - writer_start))

                self._postprocess(result, batch)

                self.logger.debug('[END] Batch total:\t{:.2f} secs'.format(time.time() - start))

                if result['global_step'] % 100 == 0:
                    self.logger.info(self._get_logline(
                        mode, result, epoch, metrics))

                if result['global_step'] % self._config['eval_every_steps'] == 0:
                    self._save(result['global_step'])
                    self.validate(validation_dataset, writer, local_init_op, epoch, metrics_fetch_dict, 'mrr')

                    if self._stopping_condition_met():
                        early_stop = True

        self.validate(validation_dataset, writer, local_init_op, epoch, metrics_fetch_dict)
        self._save(result['global_step'], export_model=True)

    # TODO: Move to abstract class
    def validate(self, validation_dataset, summary_writer,
                 local_init_op, epoch, metrics_fetch_dict, eval_metric_key=None):
        self.logger.debug('[START] Validation')
        start = time.time()
        if validation_dataset is None:
            return
        batches = validation_dataset.feature_and_label_generator()
        self._sess.run(local_init_op)
        mode = tf.estimator.ModeKeys.EVAL
        fetch_dict = self._get_fetch_dict(mode)
        for batch in batches:
            result = self._run_step(batch, fetch_dict)
            if result is None:
                continue
            metrics = self._get_metrics(metrics_fetch_dict)

        summary_writer.add_summary(
            result['summary_str'], result['global_step'])

        if eval_metric_key is not None:
            self._eval_metric_history.append(
                (result['global_step'], metrics[eval_metric_key]))

        self.logger.info(self._get_logline(mode, result, epoch, metrics))
        self.logger.debug(
            '[END] Validation took {:.2f} secs'.format(time.time() - start))

        self._save(result['global_step'], export_model=True)

    def _get_logline(self, mode, result, epoch, metrics):

        if mode == tf.estimator.ModeKeys.TRAIN:
            logline = "[TRAIN]\t"
        else:
            logline = "[EVAL]\t"

        logline += f"E: {epoch}\t"
        logline += f"S: {result['global_step']}\t"
        logline += f"CE: {result['cross_entropy_loss']:4.2f}\t"
        logline += f"TOP1: {result['top1_loss']:4.2f}\t"
        logline += f"MRR: {metrics['mrr']:4.2f}\t"
        logline += f"P: {metrics['precision']:4.2f}\t"
        logline += f"R: {metrics['recall']:4.2f}"

        return logline

    def _run_step(self, batch, fetch_dict):

        batch, user_embeddings, session_embeddings, product_embeddings = self._preprocess(
            batch)

        if batch.shape[0] == 0:
            return None

        start = time.time()
        result = self._sess.run(fetch_dict,
                                feed_dict={
                                    self._ops.features.user_embeddings: user_embeddings,
                                    self._ops.features.session_embeddings: session_embeddings,
                                    self._ops.features.product_embeddings: product_embeddings,
                                    self._ops.features.session_changed: batch['SessionChanged'],
                                    self._ops.labels: batch['LabelEmbeddingId']
                                })

        self.logger.debug('\tRun iteration:\t\t{:.2f} secs'.format(time.time() - start))

        return result

    def _get_metrics(self, fetch_dict):
        start = time.time()
        metrics = self._sess.run(fetch_dict)
        self.logger.debug('\tGet metrics:\t\t{:.2f} secs'.format(time.time() - start))
        return metrics

    def _stopping_condition_met(self):
        last_step, last_eval_metric = self._eval_metric_history[-1]
        if self._config['min_train_steps'] and last_step < self._config['min_train_steps']:
            return False
        if self._config['train_steps'] and last_step > self._config['train_steps']:
            self.logger.info('Max training steps reached, stopping early.')
            return True

        for step, eval_metric in reversed(self._eval_metric_history[:-1]):
            if last_step - step > self._config['max_steps_without_increase']:
                self.logger.info('Max steps without increase in evaluation metric reached, stopping early.')
                return True
            if eval_metric < last_eval_metric:
                return False

        return False

    # TODO: Move to abstract class
    def _restore(self, restore_embeddings):
        ckpt = tf.train.get_checkpoint_state(self._config['log_dir'] + 'checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

        if restore_embeddings:
            self._user_embeddings = json.load(open(self._config['log_dir'] + 'user_embeddings.json'))
            self._session_embeddings = json.load(open(self._config['log_dir'] + 'session_embeddings.json'))

    # TODO: Move to abstract class
    def _save(self, global_step, export_model=False):
        if not os.path.exists(self._config['log_dir'] + 'checkpoints'):
            os.makedirs(self._config['log_dir'] + 'checkpoints')

        self._saver.save(
            self._sess,
            self._config['log_dir'] + 'checkpoints/model.ckpt',
            global_step=global_step)

        if export_model:
            tf.saved_model.simple_save(
                self._sess,
                self._config['log_dir'] + 'exported_model/{}'.format(global_step),
                inputs={
                    "UserEmbeddings": self._ops.features.user_embeddings,
                    "SessionEmbeddings": self._ops.features.session_embeddings,
                    "SessionChanged": self._ops.features.session_changed,
                    "ProductEmbeddings": self._ops.features.product_embeddings
                },
                outputs={
                    "RankedPredictions": self._ops.ranked_predictions,
                    "SessionEmbeddings": self._ops.session_embeddings
                }
            )
            json.dump(self._user_embeddings, open(
                self._config['log_dir'] + 'exported_model/{}/'.format(global_step) + 'user_embeddings.json', 'w'))
            json.dump(self._session_embeddings, open(
                self._config['log_dir'] + 'exported_model/{}/'.format(global_step) + 'session_embeddings.json', 'w'))

    # TODO: Move to abstract class
    def predict(self, datapoint):
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
                     product_embeddings):

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

        predictions, session_embeddings = session_rnn(
            tf.expand_dims(product_embeddings, 1),
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

        self._ops.summaries.metrics_summaries.append(tf.summary.scalar(
            'train_metrics/mrr_at_10',
            self._ops.metrics.mrr_at_10))

        self._ops.summaries.eval_summaries.append(tf.summary.scalar(
            'eval_metrics/mrr_at_10',
            self._ops.metrics.mrr_at_10))

        (self._ops.metrics.precision_at_10,
         self._ops.metrics.precision_update_op) = tf.metrics.precision_at_k(
            labels=self._ops.labels,
            predictions=self._ops.logits,
            k=10,
            name='compute_precision')

        self._ops.summaries.metrics_summaries.append(tf.summary.scalar(
            'train_metrics/precision_at_10',
            self._ops.metrics.precision_at_10))

        self._ops.summaries.eval_summaries.append(tf.summary.scalar(
            'eval_metrics/precision_at_10',
            self._ops.metrics.precision_at_10))

        (self._ops.metrics.recall_at_10,
         self._ops.metrics.recall_update_op) = tf.metrics.recall_at_k(
            labels=self._ops.labels,
            predictions=self._ops.logits,
            k=10,
            name='compute_recall')

        self._ops.summaries.metrics_summaries.append(tf.summary.scalar(
            'train_metrics/recall_at_10',
            self._ops.metrics.recall_at_10))

        self._ops.summaries.eval_summaries.append(tf.summary.scalar(
            'eval_metrics/recall_at_10',
            self._ops.metrics.recall_at_10))

        for grad, var in self._ops.grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    self._ops.summaries.train_summaries.append(tf.summary.histogram(
                        "gradients/{}".format(var.name.replace(':', '_')),
                        grad.values))
                else:
                    self._ops.summaries.train_summaries.append(tf.summary.histogram(
                        "gradients/{}".format(var.name.replace(':', '_')),
                        grad))

            if isinstance(var, tf.IndexedSlices):
                self._ops.summaries.train_summaries.append(tf.summary.histogram(
                    "variables/{}".format(var.name.replace(':', '_')),
                    var.values))
            else:
                self._ops.summaries.train_summaries.append(tf.summary.histogram(
                    "variables/{}".format(var.name.replace(':', '_')),
                    var))

        self._ops.summaries.train_summaries.append(tf.summary.scalar('losses/cross_entropy',
                                                                     self._ops.losses.cross_entropy_loss))
        self._ops.summaries.train_summaries.append(
            tf.summary.scalar('losses/top1', self._ops.losses.top1_loss))
        self._ops.summaries.train_summaries.append(
            tf.summary.histogram('observe/top_predictions', top_predictions))
        self._ops.summaries.train_summaries.append(
            tf.summary.histogram('observe/logits', self._ops.logits))
        self._ops.summaries.train_summaries.append(
            tf.summary.histogram('observe/labels', self._ops.labels))

        if self._config['use_user_rnn']:
            self._ops.summaries.train_summaries.append(tf.summary.histogram('observe/user_embeddings',
                                                                            self._ops.user_embeddings))
        self._ops.summaries.train_summaries.append(tf.summary.histogram('observe/session_embeddings',
                                                                        self._ops.session_embeddings))

    def setup_model(self, restore=False, restore_embeddings=False):

        self._ops.global_step = tf.train.get_or_create_global_step()
        self._ops.labels = tf.placeholder(tf.int64, [None])
        self._ops.features.product_embeddings = tf.placeholder(
            tf.float32, 
            [None, 25])
        self._ops.features.session_changed = tf.placeholder(tf.bool, [None])

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
            self._ops.features.product_embeddings)

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
        elif self._config['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self._config['learning_rate'],
                momentum=self._config['momentum'])

        self._ops.grads_and_vars = [(
            tf.clip_by_norm(
                grad,
                self._config['clip_gradients_at']),
            var) for grad, var in optimizer.compute_gradients(loss)]

        self._ops.optimizer = optimizer.apply_gradients(
            self._ops.grads_and_vars,
            global_step=self._ops.global_step)

        self._setup_summaries()

        self._sess = tf.Session()
        self._saver = tf.train.Saver()

        if restore:
            self._restore(restore_embeddings)
