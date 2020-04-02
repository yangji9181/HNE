## Model: HIN2Vec

**HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning**
```
@inproceedings{fu2017hin2vec,
  title={Hin2vec: Explore meta-paths in heterogeneous information networks for representation learning},
  author={Fu, Tao-yang and Lee, Wang-Chien and Lei, Zhen},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={1797--1806},
  year={2017}
}
```

*Source: https://github.com/csiesheep/hin2vec*

### Deployment

This implementation relies on 0 external packages.

### Input

*Stage 2: Transform* prepares 1 input file stored in ```data/{dataset}```:
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{head_node_type}\t{tail_node_id}\t{tail_node_type}\t{link_type}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.