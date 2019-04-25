#!/bin/bash

pkill -f "python -m tensorboard"

rm /home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/*

python -m tensorboard.main --logdir=/home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/ &> /dev/null &

DATASET_NAME=mini_dataset && \
pipenv run python hgru4rec/hgru4rec_trainer.py \
        --train_prefix='gs://ma-muy/03_datasets/$DATASET_NAME/05_train/' \
        --eval_prefix='gs://ma-muy/03_datasets/$DATASET_NAME/06_eval/' \
        --batch_size=50 \
        --session_rnn_units=100 \
        --user_rnn_units=100 \
        --num_products=596 \
        --num_users=1089 \
        --log_dir='/home/muy/repositories/ethz-msc-thesis/artifacts/logs/hrnn_init_small_mini_dataset' \
        --epochs=100 \
        --user_dropout=0.0 \
        --session_dropout=0.1 \
        --init_dropout=0.0 \
        --learning_rate=0.4 \
        --clip_gradients_at=10