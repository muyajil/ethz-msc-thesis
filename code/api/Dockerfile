FROM python:3.6

LABEL maintainer='Mohammed Ajil <mohammed@ajil.ch>'

RUN apt-get update
EXPOSE 5000

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
                flask \
                gunicorn \
                numpy \
                requests \
                google-cloud-storage

RUN pip install https://storage.googleapis.com/dg-ml-packages/dg_ml_core.tar.gz

ADD . /session_recommendation

ENTRYPOINT ["gunicorn", "--workers", "6", "--preload", "--log-level", "INFO", "--timeout", "60", "--chdir", "/session_recommendation", "--bind", "0.0.0.0:5000", "wsgi:app"]