## Model: ConvE

**Convolutional 2D Knowledge Graph Embeddings**
```
@inproceedings{dettmers2018convolutional,
  title={Convolutional 2d knowledge graph embeddings},
  author={Dettmers, Tim and Minervini, Pasquale and Stenetorp, Pontus and Riedel, Sebastian},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```

*Source: https://github.com/TimDettmers/ConvE*

### Deployment

This implementation relies on 2 external packages:
- <a href="https://pytorch.org/">[PyTorch]</a>
- [```requirements.txt```]
```
pip install -r requirements.txt
python -m spacy download en
```

### Input

*Stage 2: Transform* prepares 1 input files stored in ```data/{dataset}```:
- ```link.dat```: Each line is formatted as ```{head_node_id}\t{link_type}\t{tail_node_id}```.

### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.