## Model: TransE

**Translating Embeddings for Modeling Multi-relational Data**
```
@inproceedings{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  booktitle={Advances in neural information processing systems},
  pages={2787--2795},
  year={2013}
}
```

*Source: https://github.com/thunlp/Fast-TransX*

### Deployment

This implementation relies on 0 external packages.

### Input

*Stage 2: Transform* prepares 3 input files stored in ```data/{dataset}```:
- ```node.dat```: Each line is formatted as ```{node_id} {node_id}```.
- ```link.dat```: Each line is formatted as ```{head_node_id} {tail_node_id} {link_type}```.
- ```rela.dat```: The first line specifies the number of link types in the targeting dataset. Each following line describes the name (string) and id (string) of a link type, which are separated by an empty space.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.