# syntax=docker/dockerfile:1

FROM python:3.10

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY /api ./api

COPY /data ./data

COPY /pickle ./pickle

COPY /dashboard ./dashboard

EXPOSE 8000

EXPOSE 8501