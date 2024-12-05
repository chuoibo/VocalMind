FROM mongo:5.0

COPY requirements.txt /tmp/requirements.txt

WORKDIR /database

RUN mkdir -p /database/logs

COPY ./src /database/src

ENV MONGO_INITDB_ROOT_USERNAME=${DATABASE_USER}
ENV MONGO_INITDB_ROOT_PASSWORD=${DATABASE_PASSWORD}

RUN pip install -r /tmp/requirements.txt


