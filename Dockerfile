# syntax=docker/dockerfile:1

FROM python:3.10

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY /api ./api

COPY /data ./data

EXPOSE 8000