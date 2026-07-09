FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

ENV GIT_SSL_NO_VERIFY=1
ENV DEBIAN_FRONTEND=noninteractive

# Update System
RUN apt update && apt upgrade -y

# Install DEPS
ARG SYS_DEPS="gnupg git curl wget ca-certificates software-properties-common net-tools"
ARG CPP_DEPS="build-essential gcc cmake pkg-config clangd"
ARG PYTHON_DEPS="python3 python3-venv python3-pip python3-dev python3-setuptools python3-tk"
ARG LIB_DEPS="libopencv-dev xtensor-dev nlohmann-json3-dev"
RUN apt update && apt install -y $SYS_DEPS $CPP_DEPS $PYTHON_DEPS $LIB_DEPS
RUN add-apt-repository universe
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install --system

# Setup Python Environment
WORKDIR /tmp
COPY requirements.txt .
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
RUN ln -sf /opt/venv/bin/python3 /opt/venv/bin/python
RUN pip install -Ur requirements.txt