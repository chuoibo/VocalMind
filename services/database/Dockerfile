ARG BASE_IMAGE=python:3.10.14-slim-bullseye
FROM ${BASE_IMAGE}

ENV HOST 0.0.0.0
ENV PORT 8000
ENV DEBUG True

WORKDIR /database

COPY . /database

RUN mkdir -p /api/logs
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD uvicorn src.main:app --host ${HOST} --port ${PORT}
