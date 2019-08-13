from dg_ml_core.file import get_paths_with_prefix, copy_file, save_to_file
from dg_ml_core.datastores import gcs_utils
import os


def get_local_path(remote_path):
    parts = remote_path.split('/')
    is_relevant = False
    path = '/models/'
    for part in parts:
        if is_relevant:
            path += (part + '/')
        if part == '04_model_artifacts':
            is_relevant = True
    return path[:-1]

client = gcs_utils.get_client(project_id='machinelearning-prod', service_account_json=None)
file_paths = get_paths_with_prefix('gs://ma-muy/04_model_artifacts/')
for remote_file_path in file_paths:
    if 'embeddings' not in remote_file_path:
        print('Downloading {}...'.format(remote_file_path))
        local_file_path = get_local_path(remote_file_path)
        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path))
        copy_file(remote_file_path, local_file_path)
