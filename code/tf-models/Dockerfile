FROM python:3.6 AS build-env

RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_core.tar.gz
RUN pip3 install google-cloud-storage

COPY ./* /
ENV GOOGLE_APPLICATION_CREDENTIALS /key.json
RUN python3 /prepare_model_artifacts.py
RUN python3 /download_artifacts.py

FROM tensorflow/serving:latest

COPY --from=build-env /models /models 