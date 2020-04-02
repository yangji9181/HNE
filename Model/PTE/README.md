## Model: PTE

**PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks**
```
@inproceedings{tang2015pte,
  title={Pte: Predictive text embedding through large-scale heterogeneous text networks},
  author={Tang, Jian and Qu, Meng and Mei, Qiaozhu},
  booktitle={Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1165--1174},
  year={2015}
}
```

*Source: https://github.com/mnqu/PTE*

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
- ```node.dat```: Each line is formatted as ```{node_id}```.
- ```link.dat```: Each line is formatted as ```{head_node_id} {tail_node_id} {link_type} {link_weight}```.
- ```type.dat```: There is only one line in this file, which describes the number of different link types in the targeting dataset.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.