## Model: HAN

**Heterogeneous Graph Attention Network**
```
@inproceedings{wang2019heterogeneous,
  title={Heterogeneous graph attention network},
  author={Wang, Xiao and Ji, Houye and Shi, Chuan and Wang, Bai and Ye, Yanfang and Cui, Peng and Yu, Philip S},
  booktitle={The World Wide Web Conference},
  pages={2022--2032},
  year={2019}
}
```

*Source: https://github.com/xiaoyuxin1002/NLAH*

### Deployment

This implementation relies on 1 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>

### Input

*Stage 2: Transform* prepares 4 input files stored in ```data/{dataset}```:
- ```node.dat```: For attributed training, each line is formatted as ```{node_id}\t{node_type}\t{node_attributes}``` where entries in ```{node_attributes}``` are separated by ```,```. For unattributed training, each line is formatted as ```{node_id}\t{node_type}```.
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{tail_node_id}\t{link_type}```.
- ```config.dat```: The first line specifies the targeting node type. The second line specifies the targeting link type. The third line specifies the information related to each link type, e.g., ```{head_node_type}\t{tail_node_type}\t{link_type}```.
- ```label.dat```: This file is only needed for semi-supervised training. Each line is formatted as ```{node_id}\t{node_label}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.