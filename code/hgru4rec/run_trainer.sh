#!/bin/bash

pkill -f "python -m tensorboard"

rm /home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/*

python -m tensorboard.main --logdir=/home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/ &> /dev/null &

python hgru4rec_trainer.py \
	--train_prefix='gs://ma-muy/baseline_dataset/train_embedded/' \
    --eval_prefix='gs://ma-muy/baseline_dataset/eval_embedded/' \
	--batch_size=10 \
	--session_rnn_units=25 \
	--user_rnn_units=50 \
	--num_products=100000 \
	--num_users=100000 \
	--train_steps=10000 \
	--log_dir='/home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/' \
	--epochs=10 \
	--user_dropout=0.0 \
	--session_dropout=0.1 \
	--init_dropout=0.0