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
            sessions_by_user_prefix):

        self.batch_size = batch_size
        self.sessions_by_user_prefix = sessions_by_user_prefix

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

            sorted_events = sorted(
                sorted_session['Events'], key=lambda z: z['Timestamp'])
            for event in sorted_events:
                yield event

            yield {'ProductId': -1, 'EmbeddingId': -1}

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
                    next_batch[idx] = \
                        (-1, {'ProductId': -1, 'EmbeddingId': -1})
                    continue
                next_event = self.get_next_event_or_none(active_users[idx])
                while next_event is None:
                    next_user = self.get_next_user_or_none(users)
                    if next_user is None:
                        tf.logging.info('There are no more new users')
                        active_users[idx] = None
                        break
                    else:
                        active_users[idx] = next_user
                        next_event = self.get_next_event_or_none(
                            active_users[idx])
                else:
                    next_batch[idx] = \
                        (active_users[idx]['UserId'], next_event)
            if len(set(map(lambda x: str(x), next_batch.values()))) == 1:
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
                        x[1]['ProductId'],
                        x[1]['EmbeddingId']),
                    next_features))

                features = list(map(
                    lambda x: (
                        x[0],
                        x[1]['ProductId'],
                        x[1]['EmbeddingId'],
                        x[1]['UserEmbeddingId']), features))

                yield features, labels
                features = next_features
            except StopIteration:
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
        epochs=1):

    dataset_wrapper = UserParallelMiniBatchDataset(
        batch_size,
        sessions_by_user_prefix)

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
        default='gs://ma-muy/baseline_dataset/train/')

    args = parser.parse_args()

    dataset = input_fn(
        args.batch_size,
        args.sessions_by_user_prefix)

    print('Datapoint:')

    for datapoint in dataset:
        print(datapoint[0])
        print(datapoint[0]['UserId'].shape[0])
        print(datapoint[1])
        break


if __name__ == "__main__":
    main()
