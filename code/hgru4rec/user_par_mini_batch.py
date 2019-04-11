"""
User-parallel mini batches dataset implementation
"""

import tensorflow as tf
from dg_ml_core.file import get_paths_with_prefix, file_exists
from dg_ml_core.collections import dict_ops
import random
import argparse


class UserParallelMiniBatchDataset(object):

    def __init__(
            self,
            batch_size,
            sessions_by_user_prefix,
            min_events_per_session,
            embedding_dict_path=None):

        self.batch_size = batch_size
        self.sessions_by_user_prefix = sessions_by_user_prefix
        self.min_events_per_session = min_events_per_session
        self.embedding_dict_path = embedding_dict_path

        if embedding_dict_path is None or not file_exists(embedding_dict_path):
            self.embedding_dict = dict()
            self.embedding_dict['User'] = dict()
            self.embedding_dict['User']['FromEmbedding'] = dict()
            self.embedding_dict['User']['ToEmbedding'] = dict()
            self.embedding_dict['Product'] = dict()
            self.embedding_dict['Product']['FromEmbedding'] = dict()
            self.embedding_dict['Product']['ToEmbedding'] = dict()
            self.next_embedding_id = dict()
            self.next_embedding_id['User'] = 0
            self.next_embedding_id['Product'] = 0
        else:
            self.embedding_dict = dict_ops.load_dict(embedding_dict_path)
            self.next_embedding_id = dict()
            self.next_embedding_id['Product'] = max(
                self.embedding_dict['Product']['ToEmbedding'].values())
            self.next_embedding_id['User'] = max(
                self.embedding_dict['User']['ToEmbedding'].values())

    def user_iterator(self):
        paths = get_paths_with_prefix(self.sessions_by_user_prefix)
        for path in paths:
            merged_shard = dict_ops.load_dict(path)
            user_ids = list(merged_shard.keys())
            random.shuffle(user_ids)
            for user_id in user_ids:
                yield user_id, merged_shard[user_id]

    def event_iterator(self, user_sessions):
        sorted_sessions = sorted(
            map(lambda x: user_sessions[x], user_sessions.keys()),
            key=lambda y: y['StartTime'])
        for sorted_session in sorted_sessions:
            if len(sorted_session['Events']) < self.min_events_per_session:
                continue

            sorted_events = sorted(
                sorted_session['Events'], key=lambda z: z['Timestamp'])
            for event in sorted_events:
                yield event['ProductId']

            yield -1

    def get_next_event_or_none(self, active_user):
        try:
            return next(active_user['Events'])
        except StopIteration:
            return None

    def get_next_user_or_none(self, users):
        try:
            user_id, user_sessions = next(users)
            return {
                'UserId': int(user_id),
                'Events': self.event_iterator(user_sessions)
            }
        except StopIteration:
            return None

    def get_embedding_id(self, item_id, embedding_type):
        if item_id < 0:
            return item_id

        if str(item_id) not in self.embedding_dict[embedding_type]['ToEmbedding']:
            self.embedding_dict[embedding_type]['ToEmbedding'][str(item_id)]\
                = self.next_embedding_id[embedding_type]

            embedding_id = self.next_embedding_id[embedding_type]

            self.embedding_dict[embedding_type]['FromEmbedding'][str(embedding_id)]\
                = item_id

            self.next_embedding_id[embedding_type] += 1

        return self.embedding_dict[embedding_type]['ToEmbedding'][str(item_id)]

    def user_parallel_batch_iterator(self):

        active_users = dict()
        users = self.user_iterator()

        # Initial fill of users
        for i in range(self.batch_size):
            active_users[i] = self.get_next_user_or_none(users)

        while True:
            next_batch = dict()
            for idx in active_users:
                if active_users[idx] is None:
                    next_batch[idx] = (-1, -1)
                    continue
                next_event = self.get_next_event_or_none(active_users[idx])
                while next_event is None:
                    next_user = self.get_next_user_or_none(users)
                    if next_user is None:
                        print('There are no more new users')
                        active_users[idx] = None
                        break
                    else:
                        active_users[idx] = next_user
                        next_event = self.get_next_event_or_none(
                            active_users[idx])
                else:
                    next_batch[idx] = \
                        (active_users[idx]['UserId'], next_event)
            if len(set(next_batch.values())) == 1:
                return
            yield list(next_batch.values())

    def feature_and_label_generator(self):
        iterator = self.user_parallel_batch_iterator()
        features = next(iterator)
        while True:
            try:
                next_features = next(iterator)
                labels = list(map(
                    lambda x: (
                        x[1],
                        self.get_embedding_id(
                            x[1],
                            'Product')),
                    next_features))

                features = list(map(
                    lambda x: (
                        x[0],
                        x[1],
                        self.get_embedding_id(x[1], 'Product'),
                        self.get_embedding_id(x[0], 'User')), features))

                yield features, labels
                features = next_features
            except StopIteration:
                if self.embedding_dict_path is not None:
                    dict_ops.save_dict(
                        self.embedding_dict_path,
                        self.embedding_dict)
                return


def generate_feature_maps(features, labels):
    features = tf.map_fn(
        lambda x: {
            'UserId': x[0],
            'ProductId': x[1],
            'EmbeddingId': x[2],
            'UserEmbeddingId': x[3]},
        features,
        dtype={
            'UserId': tf.int64,
            'ProductId': tf.int64,
            'EmbeddingId': tf.int64,
            'UserEmbeddingId': tf.int64})

    labels = tf.map_fn(
        lambda x: {'ProductId': x[0], 'EmbeddingId': x[1]},
        labels,
        dtype={'ProductId': tf.int64, 'EmbeddingId': tf.int64})
    return features, labels


def input_fn(
        batch_size,
        sessions_by_user_prefix,
        min_events_per_session,
        embedding_dict_path=None,
        epochs=1):

    dataset_wrapper = UserParallelMiniBatchDataset(
        batch_size,
        sessions_by_user_prefix,
        min_events_per_session,
        embedding_dict_path=embedding_dict_path)

    dataset = tf.data.Dataset.from_generator(
        dataset_wrapper.feature_and_label_generator,
        output_types=(tf.int64, tf.int64),
        output_shapes=(
            tf.TensorShape((batch_size, 4)),
            tf.TensorShape((batch_size, 2))))

    dataset = dataset.map(generate_feature_maps)
    dataset.repeat(epochs)

    return dataset


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument(
        '--sessions_by_user_prefix',
        type=str,
        default='gs://ma-muy/baseline_dataset/sessions_by_user/')

    parser.add_argument('--min_events_per_session', type=int, default=3)

    args = parser.parse_args()

    dataset = input_fn(
        args.batch_size,
        args.sessions_by_user_prefix,
        args.min_events_per_session)

    for datapoint in dataset:
        print(datapoint[0])
        print(datapoint[0]['UserId'].shape[0])
        print(datapoint[1])
        break


if __name__ == "__main__":
    main()
