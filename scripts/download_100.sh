#!/usr/bin/env bash
DATA_DIR=/home/lovro/PycharmProjects/RecommenderSystems/datasets
SIZE=100k
mkdir -p ${DATA_DIR}
wget http://files.grouplens.org/datasets/movielens/ml-${SIZE}.zip -O ${DATA_DIR}/ml-${SIZE}.zip
unzip ${DATA_DIR}/ml-${SIZE}.zip -d ${DATA_DIR}
