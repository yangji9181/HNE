#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
path_file="${folder}path.dat"
emb_file="${folder}emb.dat"

make

threads=5 # number of threads for training
size=50 # embedding dimension
negative=5 # number of negative samples
alpha=0.025 # initial learning rate
samples=1 # number of edges (Million) for training at each iteration
iter=500 # number of iterations

./bin/esim -model 2 -alpha ${alpha} -node ${node_file} -link ${link_file} -path ${path_file} -emb ${emb_file} -binary 0 -size ${size} -negative ${negative} -samples ${samples} -iters ${iter} -threads ${threads}