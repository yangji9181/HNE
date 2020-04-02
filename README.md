**Code Release in Progress**


# HNE
Heterogeneous Network Representation Learning: Survey, Benchmark, Evaluation, and Beyond. <a href="https://arxiv.org/pdf/2004.00216.pdf">[paper]</a>

## Citation

## Pipeline

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

We provide 10 HIN baseline implementaions: 
- 5 Proximity-Preserving Methods (```metapath2vec-ESim```, ```PTE```, ```HIN2Vec```, ```AspEm```, ```HEER```); 
- 2 Message-Passing Methods (```R-GCN```, ```HAN```); 
- 3 Relation-Learning Methods (```TransE```, ```DistMult```, ```ConvE```).

Please refer to the ```Model``` folder for more details.

### Stage 4: Evaluate

This stage evaluates the output embeddings based on specific tasks. 

Users need to specify the targeting dataset, the targeting model, and the evaluation tasks.

Please refer to the ```Evaluate``` folder for more details.