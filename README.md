# Heterogeneous Network Representation Learning: Benchmark with Data and Code

## Citation

Please cite the following work if you find the data/code useful.

```
@article{yang2020heterogeneous,
  title={Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark},
  author={Yang, Carl and Xiao, Yuxin and Zhang, Yu and Sun, Yizhou and Han, Jiawei},
  journal={TKDE},
  year={2020}
}
```

## Contact

Please contact us if you have problems with the data/code, and also if you think your work is relevant but missing from the survey.

Yuxin Xiao (yuxinx2@illinois.edu), Carl Yang (yangji9181@gmail.com)

## Guideline

### Stage 1: Data

We provide 4 HIN benchmark datasets: ```DBLP```, ```Yelp```, ```Freebase```, and ```PubMed```.

Each dataset contains:
- 3 data files (```node.dat```, ```link.dat```, ```label.dat```);
- 2 evaluation files (```link.dat.test```, ```label.dat.test```);
- 2 description files (```meta.dat```, ```info.dat```);
- 1 recording file (```record.dat```).

Please refer to the ```Data``` folder for more details.

### Stage 2: Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the targeting dataset, the targeting model, and the training settings.

Please refer to the ```Transform``` folder for more details.

### Stage 3: Model

We provide 13 HIN baseline implementaions: 
- 5 Proximity-Preserving Methods (```metapath2vec-ESim```, ```PTE```, ```HIN2Vec```, ```AspEm```, ```HEER```); 
- 4 Message-Passing Methods (```R-GCN```, ```HAN```, ```MAGNN```, ```HGT```); 
- 4 Relation-Learning Methods (```TransE```, ```DistMult```, ```ComplEx```, ```ConvE```).

Please refer to the ```Model``` folder for more details.

### Stage 4: Evaluate

This stage evaluates the output embeddings based on specific tasks. 

Users need to specify the targeting dataset, the targeting model, and the evaluation tasks.

Please refer to the ```Evaluate``` folder for more details.
