FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    cuda-command-line-tools-10-0 \
    curl \
    libsm6 \
	libfreetype6-dev \
	libhdf5-serial-dev \
	libxrender1 \
	libfontconfig1 \
	libxext6 \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    pkg-config \
    python3.6 \
    python3.6-dev \
    python3-setuptools \
    python3-pip && \
    pip3 install --upgrade pip


WORKDIR /usr/api

COPY requirements.txt .
RUN pip3 install -U --no-cache-dir -r requirements.txt

ENV PYTHONPATH ..
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH