FROM python:3.6

WORKDIR /usr/scheduler

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONPATH ..