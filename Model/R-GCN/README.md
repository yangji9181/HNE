## Model: R-GCN

**Modeling Relational Data with Graph Convolutional Networks**
```
@article{schlichtkrull2017modeling,
  title={Modeling Relational Data with Graph Convolutional Networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Berg, Rianne van den and Titov, Ivan and Welling, Max},
  journal={arXiv preprint arXiv:1703.06103},
  year={2017}
}
```

*Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>
- <a href="https://github.com/dmlc/dgl">[DGL]</a>

### Input

*Stage 2: Transform* prepares 4 input files stored in ```data/{dataset}```:
- ```node.dat```: This file is only needed for attributed training, each line is formatted as ```{node_id}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```.
- ```link.dat```: The first line specifies ```{number_of_nodes} {number_of_link_types}```. Each folloing line is formatted as ```{head_node_id} {link_type} {tail_node_id}```.
- ```label.dat```: This file is only needed for semi-supervised training. Each line is formatted as ```{node_id}\t{node_label}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.