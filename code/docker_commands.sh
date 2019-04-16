# Run tensorboard
docker run --restart always -d -v ~/logs:/logs -p 6006:6006 eu.gcr.io/machinelearning-prod/dg_ml_tensorboard:latest

# Run JupyterLab
docker run --restart always -d -v ~/ethz-msc-thesis:/ethz-msc-thesis -p 8888:8888 eu.gcr.io/machinelearning-prod/ma_muy_jupyter:latest

# Run model
docker run \
    -d \
    --runtime=nvidia \
    --cpus=$(nproc) \
    -v ~/logs:/logs \
    eu.gcr.io/machinelearning-prod/ma_muy_models:latest \
    python /code/hgru4rec/hgru4rec_trainer.py \
        --train_prefix='gs://ma-muy/baseline_dataset/train/' \
        --eval_prefix='gs://ma-muy/baseline_dataset/eval/' \
        --batch_size=50 \
        --session_rnn_units=100 \
        --user_rnn_units=100 \
        --num_products=542346 \
        --log_dir='/logs/hrnn_init_small' \
        --epochs=10 \
        --user_dropout=0.0 \
        --session_dropout=0.1 \
        --init_dropout=0.0

# Attach to logs
docker logs -f <container_name>