# FROM node:10
FROM ubuntu:18.04

RUN echo "Starting install"

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    curl \
    python3.6-dev \
    python3-pip \
    clang \
    ninja-build \
    gpg \
    gpg-agent \
    wget \
    ca-certificates \
    xz-utils \
    apt-transport-https \
    dirmngr && \
    rm -rf /var/lib/apt/lists/*

# from https://hub.docker.com/r/nikolaik/python-nodejs/dockerfile
# For some stupid reason python-minimal is a dependency of this nodejs, 
# so we end up with a python 2 install as well :(
RUN \
  echo "deb https://deb.nodesource.com/node_10.x stretch main" > /etc/apt/sources.list.d/nodesource.list && \
  wget -qO- https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add - && \
  echo "deb https://dl.yarnpkg.com/debian/ stable main" > /etc/apt/sources.list.d/yarn.list && \
  wget -qO- https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
  echo "Updating apt cache and installing node and yarn" && \
  apt-get update -qq && \
  apt-get install -y nodejs yarn && \
  pip3 install -U pip && pip install pipenv && \
  npm i -g npm@^6 && \
  rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.16.3


RUN  apt-get update -qq && \
    apt-get install -y build-essential

# Install latest CMake version from source
# RUN mkdir cmake-build && cd cmake-build && \
#     wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3.tar.gz && \
#     tar -xzvf cmake-3.14.3.tar.gz &&\
#     cd cmake-3.14.3/ && \
#     ./bootstrap && \
#     make -j4 && \
#     make install

# Install latest CMake version from a prebuilt binary
RUN mkdir cmake-build && cd cmake-build && \
    wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.14.3-Linux-x86_64.tar.gz --strip-components=1 --exclude=doc -C /usr/local/

# Done installing node.js and yarn