#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
label_file="${folder}label.dat"
emb_file="${folder}emb.dat"

size=50 # embedding dimension
negative=5 # number of negative samples
lr=0.01 # initial learning rate
dropout=0.2
regularization=0.01
grad_norm=1.0
edge_sampler="uniform"
gpu=0
num_bases=-1
num_layers=2
num_epochs=1000
graph_batch_size=20000
graph_split_size=0.5

attributed="False"
supervised="False"

python3 src/main.py --link ${link_file} --node ${node_file} --label ${label_file} --output ${emb_file} --n-hidden ${size} --negative-sample ${negative} --lr ${lr} --dropout ${dropout} --gpu ${gpu} --n-bases ${num_bases} --n-layers ${num_layers} --n-epochs ${num_epochs} --regularization ${regularization} --grad-norm ${grad_norm} --graph-batch-size ${graph_batch_size} --graph-split-size ${graph_split_size} --edge-sampler ${edge_sampler} --attributed ${attributed} --supervised ${supervised}