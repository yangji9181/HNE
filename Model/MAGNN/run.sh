#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
path_file="${folder}path.dat"
label_file="${folder}label.dat"
emb_file="${folder}emb.dat"

size=50
adim=100
nhead=8
nlayer=2
rtype="RotatE0"
dropout=0.5

nepoch=1000
batchsize=1024
sampling=100
lr=0.005
weight_decay=0.001

device="cuda"
attributed="False"
supervised="False"

python3 src/main.py --node=${node_file} --link=${link_file} --path=${path_file} --label=${label_file} --output=${emb_file} --device=${device} --hdim=${size} --adim=${adim} --nhead=${nhead} --nlayer=${nlayer} --rtype=${rtype} --dropout=${dropout} --nepoch=${nepoch} --batchsize=${batchsize} --sampling=${sampling} --lr=${lr} --weight-decay=${weight_decay} --attributed=${attributed} --supervised=${supervised}