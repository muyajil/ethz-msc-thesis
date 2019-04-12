FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

LABEL maintainer='Mohammed Ajil <mohammed.ajil@digitecgalaxus.ch>'

RUN apt-get update

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    tensorflow-gpu \
    matplotlib \
    numpy \
    pandas \
    sklearn \
    google-cloud-storage \
    google-cloud-logging \
    google-cloud-datastore

RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_core.tar.gz
RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_models.tar.gz
RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_utils.tar.gz
RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_processing.tar.gz

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir /logs

ADD ./ /code/
