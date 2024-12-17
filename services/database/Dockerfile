FROM mongo:5.0

COPY requirements.txt /tmp/requirements.txt
COPY . /database

WORKDIR /database

RUN mkdir -p /database/logs

ENV MONGO_INITDB_ROOT_USERNAME=${DATABASE_USER}
ENV MONGO_INITDB_ROOT_PASSWORD=${DATABASE_PASSWORD}

RUN pip install -r /tmp/requirements.txt


