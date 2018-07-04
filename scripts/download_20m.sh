#!/usr/bin/env bash
DATA_DIR=/home/lovro_vidovic12/Recommendation_Systems
SIZE=20m
mkdir -p ${DATA_DIR}
wget http://files.grouplens.org/datasets/movielens/ml-${SIZE}.zip -O ${DATA_DIR}/ml-${SIZE}.zip
unzip ${DATA_DIR}/ml-${SIZE}.zip -d ${DATA_DIR}
