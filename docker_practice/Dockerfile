FROM nvcr.io/nvidia/pytorch:20.11-py3

RUN apt-get update && apt-get upgrade -y && apt-get install -y vim && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install matplotlib seaborn scikit-learn scipy pandas numpy jupyter wandb apex

WORKDIR /

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME


