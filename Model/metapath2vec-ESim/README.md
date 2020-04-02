## Model: metapath2vec-ESim

**metapath2vec: Scalable Representation Learning for Heterogeneous Networks**
```
@inproceedings{dong2017metapath2vec,
  title={metapath2vec: Scalable representation learning for heterogeneous networks},
  author={Dong, Yuxiao and Chawla, Nitesh V and Swami, Ananthram},
  booktitle={Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining},
  pages={135--144},
  year={2017}
}
```

**Meta-Path Guided Embedding for Similarity Search in Large-Scale Heterogeneous Information Networks**
```
@article{shang2016meta,
  title={Meta-path guided embedding for similarity search in large-scale heterogeneous information networks},
  author={Shang, Jingbo and Qu, Meng and Liu, Jialu and Kaplan, Lance M and Han, Jiawei and Peng, Jian},
  journal={arXiv preprint arXiv:1610.09769},
  year={2016}
}
```

*Source: https://github.com/shangjingbo1226/ESim*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://www.gnu.org/software/gsl/">[GSL]</a>
```
sudo apt-get install libgsl0ldbl
```
- <a href="http://eigen.tuxfamily.org/index.php?title=Main_Page">[Eigen]</a>
```
curl https://bitbucket.org/eigen/eigen/get/3.3.3.tar.bz2  --output eigen-3.3.3.tar.gz
tar -xf eigen-3.3.3.tar.gz
mv eigen-eigen-67e894c6cd8f eigen-3.3.3
```

### Input

*Stage 2: Transform* prepares 3 input files stored in ```data/{dataset}```:
- ```node.dat```: Each line is formatted as ```{node_id} {node_type}```.
- ```link.dat```: Each line is formatted as ```{head_node_id} {tail_node_id}```.
- ```path.dat```: Each line describes a meta-path and its relative weight, which are separated by an empty space. By default, all 1-hop and 2-hop meta-paths are used and each meta-path's weight is set to the inverse of its length. Users can specify their own meta-paths and relative weights.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.