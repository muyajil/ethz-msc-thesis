# Run tensorboard
docker run --restart always -d -v ~/logs:/logs -p 6006:6006 eu.gcr.io/machinelearning-prod/dg_ml_tensorboard:latest

# Run JupyterLab
docker run --restart always -d -v ~/ethz-msc-thesis:/ethz-msc-thesis -p 8888:8888 eu.gcr.io/machinelearning-prod/ma_muy_jupyter:latest

# Run model
DATASET_NAME=mini_dataset && \
LEARNING_RATE=0.001 && \
BATCH_SIZE=50 && \
LOSS_FUNCTION=cross_entropy && \
OPTIMIZER=adam && \
\
docker restart tensorboard && \
docker pull eu.gcr.io/machinelearning-prod/ma_muy_models:combined && \
docker run \
    -d \
    --name=$LOSS_FUNCTION'_'$OPTIMIZER'_bs_'$BATCH_SIZE'_lr_'$LEARNING_RATE \
    --runtime=nvidia \
    --cpus=$(nproc) \
    -v ~/logs:/logs \
    eu.gcr.io/machinelearning-prod/ma_muy_models:combined \
    python /code/hgru4rec/hgru4rec_trainer.py \
        --train_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/05_train/' \
        --eval_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/06_eval/' \
        --batch_size=$BATCH_SIZE \
        --session_rnn_units=100 \
        --user_rnn_units=100 \
        --num_products=596 \
        --num_users=1089 \
        --log_dir='/logs/only_session_rnn/'$DATASET_NAME'/'$LOSS_FUNCTION'/'$OPTIMIZER'/bs_'$BATCH_SIZE'/lr_'$LEARNING_RATE'/' \
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
        --train_steps=200000

# Attach to logs
docker logs -f <container_name>