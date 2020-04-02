#!/bin/sh

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
rela_file="${folder}rela.dat"
emb_file="${folder}emb.dat"

make

threads=6
size=50
alpha=0.025
margin=1
epoch=400
nbatches=50

./bin/transE -entity ${node_file} -relation ${rela_file} -triplet ${link_file} -output ${emb_file} -size ${size} -out-binary 0 -epochs ${epoch} -nbatches ${nbatches} -alpha ${alpha} -margin ${margin} -threads ${threads}