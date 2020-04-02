#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
link_file="${folder}link.dat"
emb_file="${folder}emb.dat"

make

processes=2 # number of processes
size=50 # embedding dimension
negative=5 # number of negative samples
alpha=0.025 # initial learning rate
window=3 # max window length
length=100 # length of each random walk
num=10 # number of random walks starting from each node

python3 src/main.py ${link_file} ${emb_file} -d ${size} -n ${negative} -a ${alpha} -p ${processes} -w ${window} -l ${length} -k ${num}