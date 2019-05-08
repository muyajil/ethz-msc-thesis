from python:3.6

RUN apt-get update

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev

RUN pip3 install --upgrade pip pipenv

ADD ./ /code/

WORKDIR /code

RUN pipenv install --skip-lock

ENTRYPOINT ["pipenv", "run", "python", "src/train_hier_gru.py", "100", "100", "--item_key", "ItemId", "--session_key", "SessionId", "--user_key", "UserId", "--time_key", "Time", "--user_propagation_mode", "init", "--adapt", "adam", "--batch_size", "10", "--early_stopping"]