#!/bin/bash

pkill -f "python -m tensorboard"

rm /home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/*

python -m tensorboard.main --logdir=/home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/ &> /dev/null &

python hgru4rec_trainer.py \
	--sessions_by_user_prefix='gs://ma-muy/baseline_dataset/sessions_by_user/' \
	--batch_size=10 \
	--min_events_per_session=3 \
	--session_rnn_units=25 \
	--user_rnn_units=50 \
	--num_products=100000 \
	--num_users=100000 \
	--train_steps=10000 \
	--log_dir='/home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/logs/' \
	--embedding_dict_path='/home/muy/repositories/ethz-msc-thesis/artifacts/hgru4rec_test/embedding_dict.json' \
	--epochs=10
