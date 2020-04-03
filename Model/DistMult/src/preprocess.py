from __future__ import print_function
import json

import argparse
import datetime
import json
import urllib
import pickle
import os
import numpy as np
import operator
import sys

rdm = np.random.RandomState(234234)
in_path, train_path = sys.argv[1], sys.argv[2]

print('Processing dataset')

train_graph = {}
with open(in_path, 'r') as f:
    for i, line in enumerate(f):
        
        e1, rel, e2 = line[:-1].split('\t')
        e1 = e1.strip()
        e2 = e2.strip()
        rel = rel.strip()
        rel_reverse = rel+ '_reverse'

        if (e1, rel) not in train_graph:
            train_graph[(e1, rel)] = set()
        if (e2, rel_reverse) not in train_graph:
            train_graph[(e2, rel_reverse)] = set()

        train_graph[(e1, rel)].add(e2)
        train_graph[(e2, rel_reverse)].add(e1)


def write_training_graph(graph, path):
    
    with open(path, 'w') as f:
        for i, key in enumerate(graph):
            
            e1, rel = key
            entities1 = " ".join(list(graph[key]))

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = 'None'
            data_point['rel'] = rel
            data_point['rel_eval'] = 'None'
            data_point['e2_multi1'] =  entities1
            data_point['e2_multi2'] = "None"

            f.write(json.dumps(data_point)  + '\n')

            
write_training_graph(train_graph, train_path)
