import argparse
import json
import subprocess
from itertools import product

docker_command = "DATASET_NAME=mini_dataset && \
LEARNING_RATE=0.001 && \
BATCH_SIZE=50 && \
LOSS_FUNCTION=cross_entropy && \
OPTIMIZER=adam && \
MODEL_NAME=only_session_rnn && \
\
docker restart tensorboard && \
docker pull eu.gcr.io/machinelearning-prod/ma_muy_models:latest && \
docker run \
    -d \
    --name=$MODEL_NAME'_'$LOSS_FUNCTION'_'$OPTIMIZER'_bs_'$BATCH_SIZE'_lr_'$LEARNING_RATE \
    --runtime=nvidia \
    --cpus=$(nproc) \
    -v ~/logs:/logs \
    eu.gcr.io/machinelearning-prod/ma_muy_models:latest \
    python /code/hgru4rec/hgru4rec_trainer.py \
        --train_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/05_train/' \
        --eval_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/06_eval/' \
        --batch_size=$BATCH_SIZE \
        --session_rnn_units=100 \
        --user_rnn_units=100 \
        --num_products=596 \
        --num_users=1089 \
        --log_dir='/logs/'$MODEL_NAME'/'$DATASET_NAME'/'$LOSS_FUNCTION'/'$OPTIMIZER'/bs_'$BATCH_SIZE'/lr_'$LEARNING_RATE'/' \
        --epochs=1000000 \
        --user_dropout=0.0 \
        --session_dropout=0.1 \
        --init_dropout=0.0 \
        --learning_rate=$LEARNING_RATE \
        --clip_gradients_at=10 \
        --momentum=0.5 \
        --loss_function=$LOSS_FUNCTION \
        --optimizer=$OPTIMIZER \
        --use_user_rnn=False \
        --train_steps=1000000"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='grid_search_config.json')

    args = parser.parse_args()

    config = json.load(open(args.config_path))

    values = [
        config["dataset_names"],
        config["loss_functions"],
        config["optimizers"],
        config["batch_sizes"],
        config["learning_rates"]
    ]

    for (dataset_name,
            loss_function,
            optimizer,
            batch_size,
            learning_rate) in product(*values):

        # TODO: Set env vars and run command

        print(dataset_name,
              loss_function,
              optimizer,
              batch_size,
              learning_rate)

    # subprocess.run(["docker", "pull", "eu.gcr.io/machinelearning-prod/ma_muy_models:combined"])
