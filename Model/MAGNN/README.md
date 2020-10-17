## Model: MAGNN

**MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding**
```
@inproceedings{fu2020magnn,
  title={MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding},
  author={Fu, Xinyu and Zhang, Jiani and Meng, Ziqiao and King, Irwin},
  booktitle={Proceedings of The Web Conference 2020},
  pages={2331--2341},
  year={2020}
}
```

*Source: https://github.com/cynricfu/MAGNN*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>
- <a href="https://github.com/dmlc/dgl">[DGL]</a>

### Input

*Stage 2: Transform* prepares 3 input files stored in ```data/{dataset}```:
- ```node.dat```: For attributed training, each line is formatted as ```{node_id}\t{node_type}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```. For unattributed training, each line is formatted as ```{node_id}\t{node_type}```.
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{tail_node_id}\t{link_type}```.
- ```path.dat```: Each line describes a meta-path (a sequence of ```{node_type}``` separated by ```\t```). By default, all 1-hop and 2-hop meta-paths are used. Users can specify their own meta-paths.
- ```label.dat```: This file is only needed for semi-supervised training. Each line is formatted as ```{node_id}\t{node_label}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.