import argparse
import json
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='grid_search_config.json')

    args = parser.parse_args()

    config = json.load(open(args.config_path))

    subprocess.run(["docker", "pull", "eu.gcr.io/machinelearning-prod/ma_muy_models:combined"])