## Evaluate

This stage evaluates the output embeddings based on specific tasks.

Users need to specify the following parameters in ```evaluate.sh```:
- **dataset**: choose from ```DBLP```, ```Yelp```, ```Freebase```, and ```PubMed```;
- **model**: choose from ```metapath2vec-ESim```, ```PTE```, ```HIN2Vec```, ```AspEm```, ```HEER```, ```R-GCN```, ```HAN```, ```HGT```, ```TransE```, ```DistMult```, ```ConvE```;
- **attributed**: choose ```True``` for attributed training or ```False``` for unattributed training;
- **supervised**: choose ```True``` for semi-supervised training or ```False``` for unsupervised training.
- **task**: choose ```nc``` for node classification, ```lp``` for link prediction, or ```both``` for both tasks.

*Note: Only Message-Passing Methods (```R-GCN```, ```HAN```, ```HGT```) support attributed or semi-supervised training.* <br /> 
*Note: Only ```DBLP``` and ```PubMed``` contain node attributes.*

**Node Classification**: <br /> 
We train a separate linear Support Vector Machine (LinearSVC) based on the learned embeddings on 80% of the labeled nodes and predict on the remaining 20%. We repeat the process for standard five-fold cross validation and compute the average scores regarding **Macro-F1** (across all labels) and **Micro-F1** (across all nodes).

**Link Prediction**: <br /> 
We use the Hadamard function to construct feature vectors for node pairs, train a two-class LinearSVC on the 80% training links and evaluate towards the 20% held out links. We repeat the process for standard five-fold cross validation and compute the average scores regarding **AUC** (area under the ROC curve) and **MRR** (mean reciprocal rank).

Run ```bash evaluate.sh``` to complete *Stage 4: Evaluate*.

The evaluation results are stored in ```record.dat``` of the corresponding dataset. 