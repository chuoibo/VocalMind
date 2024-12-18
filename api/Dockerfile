ARG BASE_IMAGE=python:3.10.14-slim-bullseye
FROM ${BASE_IMAGE}

ENV HOST 0.0.0.0
ENV PORT 8081
ENV DEBUG True

COPY . /api
WORKDIR /api

RUN mkdir -p /api/logs
RUN pip install -U setuptools pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}
CMD uvicorn api:app --host ${HOST} --port ${PORT}
