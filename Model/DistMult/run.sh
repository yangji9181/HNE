#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
link_file="${folder}link.dat"
train_file="${folder}train.dat"
emb_file="${folder}emb.dat"

python src/preprocess.py ${link_file} ${train_file}

lr=0.003
l2=0.0
lrdecay=0.995
hiddrop=0.3
indrop=0.2
featdrop=0.2
labelsm=0.1

epochs=300
batchsize=128
loader_threads=4

size=50
shape1=5
hidsize=2048

python src/main.py --preprocess=1 --data=${dataset} --train-path=${train_file} --emb-path=${emb_file} --lr=${lr} --l2=${l2} --lr-decay=${lrdecay} --hidden-drop=${hiddrop} --input-drop=${hiddrop} --feat-drop=${featdrop} --label-smoothing=${labelsm} --epochs=${epochs} --batch-size=${batchsize} --embedding-dim=${size} --embedding-shape1=${shape1} --hidden-size=${hidsize} --loader-threads=${loader_threads}

python src/main.py --preprocess=0 --data=${dataset} --train-path=${train_file} --emb-path=${emb_file} --lr=${lr} --l2=${l2} --lr-decay=${lrdecay} --hidden-drop=${hiddrop} --input-drop=${hiddrop} --feat-drop=${featdrop} --label-smoothing=${labelsm} --epochs=${epochs} --batch-size=${batchsize} --embedding-dim=${size} --embedding-shape1=${shape1} --hidden-size=${hidsize} --loader-threads=${loader_threads}

# remove the directory storing preprocessing data
home='HOME'
rm -rf ${!home}/.data/${dataset}