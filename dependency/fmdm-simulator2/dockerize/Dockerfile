FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04


#-------------------------------------------------------------------------------
# Certificate Setup
#-------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install ca-certificates

COPY certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

RUN apt-get install apt-transport-https

ENV HTTPS_PROXY http://localhost:3128
ENV HTTP_PROXY http://localhost:3128
ENV FTP_PROXY http://localhost:3128
ENV NO_PROXY http://localhost:3128
ENV https_proxy http://localhost:3128
ENV http_proxy http://localhost:3128
ENV ftp_proxy http://localhost:3128
ENV no_proxy http://localhost:3128

# install packages
RUN apt-get update && apt-get install -y \
        dirmngr \
        gnupg2 \
        lsb-release \
	&& rm -rf /var/lib/apt/lists/*


#------------------------------------------------------------------------------
# Nvidia Drivers Installation
#------------------------------------------------------------------------------
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install --no-install-recommends -y mesa-utils \
	&& rm -rf /var/lib/apt/lists/*


#-------------------------------------------------------------------------------
# Install Locales
#-------------------------------------------------------------------------------
# install locales and set locale to utf-8
RUN apt-get update && apt-get install -y locales \
	&& rm -rf /var/lib/apt/lists/*
RUN locale-gen "en_US.UTF-8"
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


#-------------------------------------------------------------------------------
# Install Project Dependencies
#-------------------------------------------------------------------------------
# install and upgrade pip
RUN apt-get update && apt-get install -y python3-venv python3-pip \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip

# install project and dependencies
WORKDIR /usr/src/app
COPY app/requirements.txt .
RUN pip3 install -r requirements.txt
COPY app/. .
RUN pip3 install -e .
RUN cd ../ && rm -r app/

# install vim to allow changes in-docker
RUN apt-get update && apt-get install -y vim \
	&& rm -rf /var/lib/apt/lists/*

# update packages
RUN apt-get update && apt-get upgrade -y \
	&& rm -rf /var/lib/apt/lists/*

# setup container
CMD ["/bin/bash"]
