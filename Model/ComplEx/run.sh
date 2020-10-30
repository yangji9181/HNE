#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
rela_file="${folder}rela.dat"
emb_file="${folder}emb.dat"

size=50
regul_rate=1.0
batch_size=1024
train_times=50
alpha=0.5
opt_method="adagrad"
if_gpu=1

g++ src/base/Base.cpp -fPIC -shared -o src/base/Base.so -pthread -O3 -march=native

python3 src/main.py --node=${node_file} --link=${link_file} --rela=${rela_file} --output=${emb_file} --dim=${size} --regul_rate=${regul_rate} --batch_size=${batch_size} --train_times=${train_times} --alpha=${alpha} --opt_method=${opt_method} --if_gpu=${if_gpu}