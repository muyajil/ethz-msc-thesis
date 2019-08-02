"""
User-parallel mini batches dataset implementation
"""

import tensorflow as tf
from dg_ml_core.file import get_paths_with_prefix, file_exists
from dg_ml_core.collections import dict_ops
import random
import argparse
from dg_ml_core.datastores import gcs_utils
import pandas as pd
import requests


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

    def _user_iterator(self):
        paths = get_paths_with_prefix(
            self.sessions_by_user_prefix,
            gcs_client=self.client)
        random.shuffle(paths)
        for path in paths:
            merged_shard = None
            while merged_shard is None:
                try:
                    merged_shard = dict_ops.load_dict(path, gcs_client=self.client)
                except requests.exceptions.ChunkedEncodingError:
                    pass
            user_ids = list(merged_shard.keys())
            random.shuffle(user_ids)
            for user_id in user_ids:
                yield user_id, merged_shard[user_id]

    def _event_iterator(self, user_sessions):
        sorted_session_ids = sorted(user_sessions.keys(),
                                    key=lambda y: user_sessions[y]['StartTime'])
        for session_id in sorted_session_ids:

            sorted_events = sorted(
                user_sessions[session_id]['Events'], key=lambda z: z['Timestamp'])
            for idx, event in enumerate(sorted_events):
                if idx == 0:
                    event['SessionChanged'] = 1
                else:
                    event['SessionChanged'] = 0

                if idx == (len(sorted_events) - 1):
                    event['LastSessionEvent'] = 1
                else:
                    event['LastSessionEvent'] = 0

                event['SessionId'] = int(session_id)
                yield event

    def _get_next_event_or_none(self, active_user):
        try:
            return next(active_user['Events'])
        except StopIteration:
            return None

    def _get_next_user_or_none(self, users):
        try:
            user_id, user_sessions = next(users)
            return {
                'UserId': int(user_id),
                'Events': self._event_iterator(user_sessions)
            }
        except StopIteration:
            return None

    def _user_parallel_batch_iterator(self):

        active_users = dict()
        users = self._user_iterator()

        # Initial fill of users
        for i in range(self.batch_size):
            active_users[i] = self._get_next_user_or_none(users)

        while True:
            next_batch = dict()
            for idx in active_users:
                # Catch no more users case
                if active_users[idx] is None:
                    break

                # There are still users available
                else:
                    next_event = self._get_next_event_or_none(active_users[idx])
                    while next_event is None:
                        next_user = self._get_next_user_or_none(users)
                        if next_user is None:
                            active_users[idx] = None
                            break
                        else:
                            active_users[idx] = next_user
                            next_event = self._get_next_event_or_none(
                                active_users[idx])
                            if next_event is not None:
                                next_event['UserChanged'] = 1

                    if active_users[idx] is None:
                        break

                    else:
                        if 'UserChanged' not in next_event:
                            next_event['UserChanged'] = 0
                        next_event['UserId'] = active_users[idx]['UserId']
                        next_batch[idx] = next_event

            if None in active_users.values():
                break

            yield pd.DataFrame(list(next_batch.values()))

    def feature_and_label_generator(self):
        iterator = self._user_parallel_batch_iterator()

        features = next(iterator)
        while True:
            next_features = next(iterator)

            features['LabelEmbeddingId'] = next_features['EmbeddingId']

            yield features
            features = next_features


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument(
        '--sessions_by_user_prefix',
        type=str,
        default='gs://ma-muy/baseline_dataset/train/')

    args = parser.parse_args()

    # dataset = input_fn(
    #     args.batch_size,
    #     args.sessions_by_user_prefix)

    # print('Datapoint:')

    # for datapoint in dataset:
    #     print(datapoint[0])
    #     print(datapoint[0]['UserId'].shape[0])
    #     print(datapoint[1])
    #     break


if __name__ == "__main__":
    main()
