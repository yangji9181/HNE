## Model: HEER

**Easing Embedding Learning by Comprehensive Transcription of Heterogeneous Information Networks**
```
@inproceedings{shi2018easing,
  title={Easing embedding learning by comprehensive transcription of heterogeneous information networks},
  author={Shi, Yu and Zhu, Qi and Guo, Fang and Zhang, Chao and Han, Jiawei},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2190--2199},
  year={2018}
}
```

*Source: https://github.com/GentleZhu/HEER*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://www.gnu.org/software/gsl/">[GSL]</a>
```
sudo apt-get install libgsl0ldbl
```
- <a href="https://pytorch.org/">[PyTorch]</a>

### Input

*Stage 2: Transform* prepares 2 input files stored in ```data/{dataset}```:
- ```link.dat```: Each line is formatted as ```{head_node_type:head_node_id} {tail_node_type:tail_node_id} {link_weight} {link_type:if_directed}```. If the link is directed, then ```{if_directed}```=```d```, otherwise ```if_directed```=```u```.
- ```config.dat```: The first line specifies a pair of node types for each link type, e.g., ```[[{head_node_type} [tail_node_type}], [{head_node_type} [tail_node_type}]]```. The second line specifies the node types, e.g., ```['{node_type}', '{node_type}']```. The third line specifies the link types, e.g., ```['{link_type:if_directed}', {link_type:if_directed}]```. The fourth line specifies the ```if_directed``` status of each link type, e.g., ```[{if_directed}, {if_directed}]```. The entries with the same index in the first, third, and fourth lines correspond to the same link type.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.