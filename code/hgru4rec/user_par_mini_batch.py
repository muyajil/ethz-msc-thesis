"""
User-parallel mini batches dataset implementation
"""

import tensorflow as tf
from dg_ml_core.file import get_paths_with_prefix, file_exists
from dg_ml_core.collections import dict_ops
import random
import argparse
from dg_ml_core.datastores import gcs_utils


class UserParallelMiniBatchDataset(object):

    def __init__(
            self,
            batch_size,
            sessions_by_user_prefix):

        self.batch_size = batch_size
        self.sessions_by_user_prefix = sessions_by_user_prefix
        self.client = gcs_utils.get_client(
            project_id='machinelearning-prod',
            service_account_json=None)
        self.epoch = 0

    def user_iterator(self):
        paths = get_paths_with_prefix(
            self.sessions_by_user_prefix,
            gcs_client=self.client)
        random.shuffle(paths)
        for path in paths:
            merged_shard = dict_ops.load_dict(path, gcs_client=self.client)
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
            for idx, event in enumerate(sorted_events):
                if idx == 0:
                    event['SessionChanged'] = 1
                else:
                    event['SessionChanged'] = 0

                if idx == (len(sorted_events) - 1):
                    event['LastSessionEvent'] = 1
                else:
                    event['LastSessionEvent'] = 0
                yield event

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
                # Catch no more users case
                if active_users[idx] is None:
                    break

                # There are still users available
                else:
                    next_event = self.get_next_event_or_none(active_users[idx])
                    while next_event is None:
                        next_user = self.get_next_user_or_none(users)
                        if next_user is None:
                            active_users[idx] = None
                            break
                        else:
                            active_users[idx] = next_user
                            next_event = self.get_next_event_or_none(
                                active_users[idx])
                            if next_event is not None:
                                next_event['UserChanged'] = 1

                    if active_users[idx] is None:
                        break

                    else:
                        if 'UserChanged' not in next_event:
                            next_event['UserChanged'] = 0

                        next_batch[idx] = \
                            (active_users[idx]['UserId'], next_event)

            if None in active_users.values():
                break

            yield list(next_batch.values())

    def feature_and_label_generator(self):
        iterator = self.user_parallel_batch_iterator()
        features = next(iterator)
        while True:

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
                    x[1]['UserEmbeddingId'],
                    x[1]['SessionChanged'],
                    x[1]['LastSessionEvent'],
                    x[1]['UserChanged'],
                    self.epoch), features))

            yield features, labels
            features = next_features

        self.epoch += 1


def generate_feature_maps(features, labels):
    features = tf.map_fn(
        lambda x: {
            'UserId': x[0],
            'ProductId': x[1],
            'EmbeddingId': x[2],
            'UserEmbeddingId': x[3],
            'SessionChanged': x[4],
            'LastSessionEvent': x[5],
            'UserChanged': x[6],
            'Epoch': x[7]},
        features,
        dtype={
            'UserId': tf.int64,
            'ProductId': tf.int64,
            'EmbeddingId': tf.int64,
            'UserEmbeddingId': tf.int64,
            'SessionChanged': tf.int64,
            'LastSessionEvent': tf.int64,
            'UserChanged': tf.int64,
            'Epoch': tf.int64})

    labels = tf.map_fn(
        lambda x: {'ProductId': x[0], 'EmbeddingId': x[1]},
        labels,
        dtype={'ProductId': tf.int64, 'EmbeddingId': tf.int64})
    return features, labels


def input_fn(
        batch_size,
        sessions_by_user_prefix,
        epochs=None):

    dataset_wrapper = UserParallelMiniBatchDataset(
        batch_size,
        sessions_by_user_prefix)

    dataset = tf.data.Dataset.from_generator(
        dataset_wrapper.feature_and_label_generator,
        output_types=(tf.int64, tf.int64),
        output_shapes=(
            tf.TensorShape((batch_size, 8)),
            tf.TensorShape((batch_size, 2))))

    dataset = dataset.map(generate_feature_maps)
    dataset = dataset.repeat(epochs)

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
