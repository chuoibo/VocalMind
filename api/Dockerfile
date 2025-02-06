ARG BASE_IMAGE=python:3.10.14-slim-bullseye
FROM ${BASE_IMAGE}

ENV HOST 0.0.0.0
ENV PORT 8081
ENV DEBUG True

COPY . /api
WORKDIR /api

RUN mkdir -p /api/logs
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    pulseaudio && \
    rm -rf /var/lib/apt/lists/* 
RUN pip install -U setuptools pyaudio 
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}
CMD uvicorn api:app --host ${HOST} --port ${PORT}
