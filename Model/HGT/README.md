## Model: HGT

**Heterogeneous Graph Transformer**
```
@inproceedings{hu2020heterogeneous,
  title={Heterogeneous graph transformer},
  author={Hu, Ziniu and Dong, Yuxiao and Wang, Kuansan and Sun, Yizhou},
  booktitle={Proceedings of The Web Conference 2020},
  pages={2704--2710},
  year={2020}
}
```

*Source: https://github.com/acbull/pyHGT*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>
- <a href="https://github.com/rusty1s/pytorch_geometric">[PyTorch Geometric]</a>

### Input

*Stage 2: Transform* prepares 3 input files stored in ```data/{dataset}```:
- ```node.dat```: For attributed training, each line is formatted as ```{node_id}\t{node_type}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```. For unattributed training, each line is formatted as ```{node_id}\t{node_type}```.
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{tail_node_id}\t{link_type}```.
- ```label.dat```: This file is only needed for semi-supervised training. The first line specifies the labeled node type and the number of label types. Each following line is formatted as ```{node_id}\t{node_label}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.