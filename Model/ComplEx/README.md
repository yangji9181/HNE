## Model: ComplEx

**Complex Embeddings for Simple Link Prediction**
```
@inproceedings{trouillon2016complex,
  title={Complex embeddings for simple link prediction},
  author={Trouillon, Th{\'e}o and Welbl, Johannes and Riedel, Sebastian and Gaussier, {\'E}ric and Bouchard, Guillaume},
  year={2016},
  organization={International Conference on Machine Learning (ICML)}
}
```

*Source: https://github.com/thunlp/OpenKE*

### Deployment

This implementation relies on 1 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>

### Input

*Stage 2: Transform* prepares 3 input files stored in ```data/{dataset}```:
- ```node.dat```: The first line specifies the number of nodes. Each following line is formatted as ```{node_id}\t{node_id}```.
- ```link.dat```: The first line specifies the number of links. Each following line is formatted as ```{head_node_id} {tail_node_id} {link_type}```.
- ```rela.dat```: The first line specifies the number of link types. Each following line is formatted as ```{link_name}\t{link_type}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.