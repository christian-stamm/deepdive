# docker/base/Dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        clangd \
        cmake \
        curl \
        ffmpeg \
        gcc \
        gdb \
        gdb-multiarch \
        git \
        git-lfs \
        gnupg \
        libopencv-dev \
        libtool \
        locales \
        net-tools \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        python3-venv \
        software-properties-common \
        texinfo \
        unzip \
        wget \
        xtensor-dev \
        xz-utils \
    && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

# Setup Python Environment
WORKDIR /tmp
COPY requirements.txt .
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN python3 -m venv ${VIRTUAL_ENV} && \
    ln -sf ${VIRTUAL_ENV}/bin/python3 ${VIRTUAL_ENV}/bin/python && \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -U pip wheel setuptools && \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -Ur requirements.txt