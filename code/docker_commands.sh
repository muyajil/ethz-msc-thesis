# Run tensorboard
docker run --restart always -d -v ~/logs:/logs -p 6006:6006 eu.gcr.io/machinelearning-prod/dg_ml_tensorboard:latest

# Run JupyterLab
docker run --restart always -d -v ~/ethz-msc-thesis:/ethz-msc-thesis -p 8888:8888 eu.gcr.io/machinelearning-prod/ma_muy_jupyter:latest

# Run model
docker run \
    --rm \
    --runtime=nvidia \
    --cpus=$(nproc) \
    -v ~/logs:/logs \
    eu.gcr.io/machinelearning-prod/ma_muy_image:latest \
    python /code/hgru4rec/hgru4rec_trainer.py \
        --sessions_by_user_prefix='gs://ma-muy/baseline_dataset/sessions_by_user/' \
        --batch_size=10 \
        --min_events_per_session=3 \
        --session_rnn_units=100 \
        --user_rnn_units=100 \
        --num_products=1258092 \
        --num_users=1211522 \
        --log_dir='/logs/baseline_run' \
        --embedding_dict_path='gs://ma-muy/embedding_dict.json' \
        --epochs=10 \
        --num_partitions=1000