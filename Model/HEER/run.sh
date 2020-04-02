#!/bin/bash

dataset="PubMed"
folder="data/${dataset}/"
link_file="${folder}link.dat"
config_file="${folder}config.dat"
pre_file="${folder}pre.dat"
emb_file="${folder}emb.dat"

size=50 # embedding dimension
half=2

# pretrain LINE
make
./bin/line -train ${link_file} -output ${pre_file} -size $((size/half)) -order 2 -negative 5 -samples 1 -alpha 0.025 -threads 5 -debug 2 -binary 0

rescale=0.1
lr=10
lrr=10
batch_size=5000
iter=5

mkdir "${folder}temp/"
mkdir "${folder}model/"
mkdir "${folder}log/"

# build graph
python2 src/main.py --build-graph=1 --link=${link_file} --config=${config_file} --temp-dir="${folder}temp/" --data-name=${dataset} 

# train embedding
python2 src/main.py --build-graph=0 --link=${link_file} --config=${config_file} --pre-train-path=${pre_file} --output=${emb_file} --temp-dir="${folder}/temp/" --model-dir="${folder}/model/" --log-dir="${folder}/log/" --data-name=${dataset} --dimensions=${size} --gpu=0 --rescale=${rescale} --lr=${lr} --lrr=${lrr} --batch-size=${batch_size} --iter=${iter} --dump-timer=100 --op=1 --map-func=0 --fine-tune=0

# remove temp folder, model folder, log folder
rm -rf "${folder}temp/"
rm -rf "${folder}model/"
rm -rf "${folder}log/"