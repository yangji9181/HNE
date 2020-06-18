#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
label_file="${folder}label.dat"
emb_file="${folder}emb.dat"

size=50
nhead=5
nlayer=3
dropout=0.2
sample_depth=6
sample_width=128

nepoch=100
npool=4
nbatch=32
repeat=2
batch_size=256
clip=0.25
cuda=0

attributed="False"
supervised="False"

python3 src/main.py --node=${node_file} --link=${link_file} --label=${label_file} --output=${emb_file} --cuda=${cuda} --n_hid=${size} --n_heads=${nhead} --n_layers=${nlayer} --dropout=${dropout} --sample_depth=${sample_depth} --sample_width=${sample_width} --n_epoch=${nepoch} --n_pool=${npool} --n_batch=${nbatch} --repeat=${repeat} --batch_size=${batch_size} --clip=${clip} --attributed=${attributed} --supervised=${supervised}