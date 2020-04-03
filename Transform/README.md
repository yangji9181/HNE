## Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the following parameters in ```transform.sh```:
- **dataset**: choose from ```DBLP```, ```Yelp```, ```Freebase```, and ```PubMed```;
- **model**: choose from ```metapath2vec-ESim```, ```PTE```, ```HIN2Vec```, ```AspEm```, ```HEER```, ```R-GCN```, ```HAN```, ```TransE```, ```DistMult```, ```ConvE```;
- **attributed**: choose ```True``` for attributed training or ```False``` for unattributed training;
- **supervised**: choose ```True``` for semi-supervised training or ```False``` for unsupervised training.

*Note: Only Message-Passing Methods (```R-GCN```, ```HAN```) support attributed or semi-supervised training.*
*Note: Only ```DBLP``` and ```PubMed``` contain node attributes.*