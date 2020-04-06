#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
config_file="${folder}config.dat"
link_file="${folder}link.dat"
label_file="${folder}label.dat"
emb_file="${folder}emb.dat"

meta="1,2,4,8" # Choose the meta-paths used for training. Suppose the targeting node type is 1 and link type 1 is between node type 0 and 1, then meta="1" means that we use meta-paths "101".
size=50
nhead="8"
dropout=0.4
neigh_por=0.6
lr=0.005
weight_decay=0.0005
batch_size=256
epochs=5 #00
device="cuda"

attributed="False"
supervised="False"

python3 src/main.py --node=${node_file} --link=${link_file} --config=${config_file} --label=${label_file} --output=${emb_file} --device=${device} --meta=${meta} --size=${size} --nhead=${nhead} --dropout=${dropout} --neigh-por=${neigh_por} --lr=${lr} --weight-decay=${weight_decay} --batch-size=${batch_size} --epochs=${epochs} --attributed=${attributed} --supervised=${supervised}