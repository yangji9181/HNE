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



### Input



### Run

Users need to specify the targeting dataset and the set of training parameters in ```run.sh```. <br /> 
Run ```bash run.sh``` to start training.

### Output

This implementation generates 1 output file stored in ```data/{dataset}```:
- ```emb.dat```: The first line specifies the parameters used in training. Each following line describes the id and the embeddings of a node. The id and the embeddings are separated by ```\t```. Entries in the embeddings are separated by ``` ```.