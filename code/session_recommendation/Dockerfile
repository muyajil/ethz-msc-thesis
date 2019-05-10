FROM python:3.6

LABEL maintainer='Mohammed Ajil <mohammed@ajil.ch>'

RUN apt-get update
EXPOSE 5000

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
                flask \
                gunicorn \
                tensorflow

ADD . /app
ENTRYPOINT ["gunicorn", "--preload", "--chdir", "/app", "--bind", "0.0.0.0:5000", "wsgi:app"]