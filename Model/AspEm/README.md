## Model: AspEm

**AspEm: Embedding Learning by Aspects in Heterogeneous Information Networks**
```
@inproceedings{shi2018aspem,
  title={Aspem: Embedding learning by aspects in heterogeneous information networks},
  author={Shi, Yu and Gui, Huan and Zhu, Qi and Kaplan, Lance and Han, Jiawei},
  booktitle={Proceedings of the 2018 SIAM International Conference on Data Mining},
  pages={144--152},
  year={2018},
  organization={SIAM}
}
```

*Source: https://github.com/ysyushi/aspem*

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
- ```node.dat```: Each line is formatted as ```{node_type:node_id} {node_type}```.
- ```link.dat```: Each line is formatted as ```{head_node_type:head_node_id} {head_node_type} {tail_node_type:tail_node_type} {tail_node_type} {link_weight} {link_type}```.
- ```type.dat```: The first line specifies the targeting node type. The second line specifies the number of node types.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.