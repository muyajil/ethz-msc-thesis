# Run tensorboard
docker run --restart always -d -v ~/logs:/logs -p 6006:6006 eu.gcr.io/machinelearning-prod/dg_ml_tensorboard:latest

# Run JupyterLab
docker run --restart always -d -v ~/ethz-msc-thesis:/ethz-msc-thesis -p 8888:8888 eu.gcr.io/machinelearning-prod/ma_muy_jupyter:latest

# Run model
DATASET_NAME=midi_dataset && \
LEARNING_RATE=0.001 && \
BATCH_SIZE=50 && \
LOSS_FUNCTION=top_1 && \
OPTIMIZER=adam && \
MODEL_NAME=with_user_medium && \
\
docker pull eu.gcr.io/machinelearning-prod/ma_muy_models:latest && \
docker run \
    -d \
    --name=$MODEL_NAME'_'$DATASET_NAME'_'$LOSS_FUNCTION'_'$OPTIMIZER'_bs_'$BATCH_SIZE'_lr_'$LEARNING_RATE \
    --runtime=nvidia \
    --cpus=$(nproc) \
    -v ~/logs:/logs \
    eu.gcr.io/machinelearning-prod/ma_muy_models:latest \
    python /code/hgru4rec/hgru4rec_trainer.py \
        --train_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/05_train/' \
        --eval_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/06_eval/' \
        --num_predictions=10 \
        --batch_size=$BATCH_SIZE \
        --session_rnn_units=250 \
        --user_rnn_units=250 \
        --num_products=47859 \
        --num_users=13395 \
        --log_dir='/logs/'$MODEL_NAME'/'$DATASET_NAME'/'$LOSS_FUNCTION'/'$OPTIMIZER'/bs_'$BATCH_SIZE'/lr_'$LEARNING_RATE'/' \
        --user_dropout=0.0 \
        --session_dropout=0.1 \
        --init_dropout=0.0 \
        --learning_rate=$LEARNING_RATE \
        --clip_gradients_at=10 \
        --momentum=0.5 \
        --loss_function=$LOSS_FUNCTION \
        --optimizer=$OPTIMIZER \
        --use_user_rnn=True \
        --min_train_steps=100000 \
        --eval_every_steps=20000 \


        --epochs=10 \
        --train_steps=200000

# Attach to logs
docker logs -f <container_name>

DATASET_NAME=midi_dataset && \
LEARNING_RATE=0.001 && \
BATCH_SIZE=50 && \
LOSS_FUNCTION=top_1 && \
OPTIMIZER=adam && \
MODEL_NAME=test_model && \
\
docker pull eu.gcr.io/machinelearning-prod/ma_muy_models:latest && \
docker run \
    -d \
    --name=test \
    --runtime=nvidia \
    --cpus=$(nproc) \
    -v ~/logs:/logs \
    eu.gcr.io/machinelearning-prod/ma_muy_models:latest \
    python /code/hgru4rec/hgru4rec_trainer.py \
        --train_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/05_train/' \
        --eval_prefix='gs://ma-muy/03_datasets/'$DATASET_NAME'/06_eval/' \
        --num_predictions=10 \
        --batch_size=$BATCH_SIZE \
        --session_rnn_units=250 \
        --user_rnn_units=250 \
        --num_products=47859 \
        --num_users=13395 \
        --log_dir='/logs/test' \
        --user_dropout=0.0 \
        --session_dropout=0.1 \
        --init_dropout=0.0 \
        --learning_rate=$LEARNING_RATE \
        --clip_gradients_at=10 \
        --momentum=0.5 \
        --loss_function=$LOSS_FUNCTION \
        --optimizer=$OPTIMIZER \
        --min_train_steps=100000 \
        --eval_every_steps=20000