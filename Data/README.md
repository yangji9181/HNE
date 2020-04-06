## Data

We provide 4 HIN benchmark datasets: ```DBLP```, ```Yelp```, ```Freebase```, and ```PubMed```.

Users can retrieve them <a href="https://drive.google.com/open?id=1Pkbl2wkwAXVRYrUWKpa1C4YQdjl_oIu2">here</a> and unzip the downloaded file to the current folder.

The statistics of each dataset are as follows.

**Dataset** | #node types | #nodes | #link types | #links | #attributes | #attributed nodes | #label types | #labeled nodes
--- | --- | --- | --- | --- | --- | --- | --- | ---
**DBLP** | 4 | 1,989,077 | 6 | 275,940,913 | 300 | ALL | 13 | 618
**Yelp** | 4 | 82,465 | 4 | 30,542,675 | N/A | N/A | 16 | 7,417
**Freebase** | 8 | 12,164,758 | 36 | 62,982,566 | N/A | N/A | 8 | 47,190
**PubMed** | 4 | 63,109 | 10 | 244,986 | 200 | ALL | 8 | 454

Each dataset contains:
- 3 data files (```node.dat```, ```link.dat```, ```label.dat```);
- 2 evaluation files (```link.dat.test```, ```label.dat.test```);
- 2 description files (```meta.dat```, ```info.dat```);
- 1 recording file (```record.dat```).

### node.dat

- In each line, there are 4 elements (```node_id```, ```node_name```, ```node_type```, ```node_attributes```) separated by ```\t```.
- ```Node_name``` is in Line ```node_id```.
- In ```node_name```, empty space (``` ```) is replaced by underscore (```_```).
- In ```node_attributes```, attributes are separated by comma (```,```).
- ```DBLP``` and ```PubMed``` contain attributes, while ```Freebase``` and ```Yelp``` do not contain attributes.

### link.dat

- In each line, there are 4 elements (```node_id```, ```node_id```, ```link_type```, ```link_weight```) separated by ```\t```.
- All links are directed. Each node is connected by at least one link.

### label.dat

- In each line, there are 4 elements (```node_id```, ```node_name```, ```node_type```, ```node_label```) separated by ```\t```.
- All labeled nodes are of the same ```node_type```.
- For ```DBLP```, ```Freebase```, and ```PubMed```, each labeled node only has one label. For ```Yelp```, each labeled node has one or multiple labels separated by ```,```.
- For unsupervised training, ```label.dat``` and ```label.dat.test``` are merged for five-fold cross validation. For semi-supervised training, ```label.dat``` is used for training and ```label.dat.test``` is used for testing.

### link.dat.test

- In each line, there are 3 elements (```node_id```, ```node_id```, ```link_status```) separated by ```\t```.
- For ```link_status```, ```1``` indicates a positive link and ```0``` indicates a negative link.
- Positive and negative links are of the same ```link_type```.
- Number of positive links = Number of negative links = One fourth of the number of real links of the same type in ```label.dat```.

### label.dat.test

- In each line, there are 4 elements (```node_id```, ```node_name```, ```node_type```, ```node_label```) separated by ```\t```.
- All labeled nodes are of the same ```node_type```.
- Number of labeled nodes in ```label.dat.test``` = One fourth of the number of labeled nodes in ```label.dat```.
- For ```DBLP```, ```Freebase```, and ```PubMed```, each labeled node only has one label. For ```Yelp```, each labeled node has one or multiple labels separated by ```,```.
- For unsupervised training, ```label.dat``` and ```label.dat.test``` are merged for five-fold cross validation. For semi-supervised training, ```label.dat``` is used for training and ```label.dat.test``` is used for testing.

### meta.dat

- This file describes the number of instances of each node type, link type, and label type in the corresponding dataset.

### info.dat

- This file describes the meaning of each node type, link type, and label type in the corresponding dataset.

### record.dat

- In each paragraph, the first line tells the model and evaluation settings, the second line tells the set of training parameters, and the third line tells the evaluation results.