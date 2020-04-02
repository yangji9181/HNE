#!/bin/sh

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
type_file="${folder}type.dat"
emb_file="${folder}emb.dat"

make

threads=4 # number of threads for training
size=50 # embedding dimension
negative=5 # number of negative samples
alpha=0.025 # initial learning rate
sample=500 # number of training samples (Million)

./bin/pte -nodes ${node_file} -words ${node_file} -hin ${link_file} -type ${type_file} -output ${emb_file} -binary 0 -size ${size} -negative ${negative} -samples ${sample} -alpha ${alpha} -threads ${threads}
