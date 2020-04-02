#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
type_file="${folder}type.dat"
baseinc_file="${folder}baseinc.dat"
allinc_file="${folder}allinc.dat"
emb_file="${folder}emb.dat"

make

threads=4 # number of threads for training
size=50 # embedding dimension
negative=5 # number of negative samples
alpha=0.025 # initial learning rate
sample=400 # number of training samples (Million)

python3 src/main.py -nodes ${node_file} -links ${link_file} -type ${type_file} -output ${emb_file} -baseinc ${baseinc_file} -allinc ${allinc_file} -binary 0 -size ${size} -negative ${negative} -samples ${sample} -alpha ${alpha} -threads ${threads}