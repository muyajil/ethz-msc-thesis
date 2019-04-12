FROM python:3

LABEL maintainer='Mohammed Ajil <mohammed.ajil@digitecgalaxus.ch>'

RUN apt-get update

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    tensorflow \
    matplotlib \
    numpy \
    pandas \
    sklearn \
    google-cloud-storage \
    google-cloud-logging \
    google-cloud-datastore \
    google-cloud-bigquery \
    jupyterlab

RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_core.tar.gz
RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_models.tar.gz
RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_utils.tar.gz
RUN pip3 install https://storage.googleapis.com/dg-ml-packages/dg_ml_processing.tar.gz

RUN jupyter notebook --generate-config

RUN mkdir /ethz-msc-thesis

EXPOSE 8888

ENTRYPOINT ["jupyter-lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir", "/ethz-msc-thesis"]
