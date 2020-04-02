import gc
import sys
import pickle
import numpy as np

class HinLoader(object):
	"""docstring for HinLoader"""
	def __init__(self, arg):
		self.in_mapping = dict()
		self.out_mapping = dict()
		self.input = list()
		self.output = list()
		self.arg = arg
		self.edge_stat = [0] * len(self.arg['edge_types'])
		for k in arg['types']:
			self.in_mapping[k] = dict()
			self.out_mapping[k] = dict()

	def inNodeMapping(self, key, type):
		if key not in self.in_mapping[type]:
			self.out_mapping[type][len(self.in_mapping[type])] = key
			self.in_mapping[type][key] = len(self.in_mapping[type])

		return self.in_mapping[type][key]

	def readHin(self, _edge_types):
		#num_nodes = defaultdict(int)          
		with open(self.arg['graph']) as INPUT:
			for index, line in enumerate(INPUT):
				if index%5000000==0: 
					print('finish readHin {}'.format(index))
					sys.stdout.flush()
				edge = line.strip().split(' ')
				edge_type = _edge_types.index(edge[-1])
				node_a = edge[0].split(':')
				node_b = edge[1].split(':')
				node_a_type = self.arg['types'].index(node_a[0])
				node_b_type = self.arg['types'].index(node_b[0])
				#assert edge_type != 11
				self.edge_stat[edge_type] += 1
				assert [node_a_type, node_b_type] == self.arg['edge_types'][edge_type][:2]
				self.input.append([edge_type, self.inNodeMapping(node_a[1], node_a[0])])
				self.output.append([self.arg['types'].index(node_b[0]), self.inNodeMapping(node_b[1], node_b[0])])
	
	def encode(self, dump_path, split=10):
		self.encoder = dict()
		offset = 0
		for k in self.arg['types']:
			self.encoder[k] = offset
			offset += len(self.in_mapping[k])
		print('dump mapping')
		sys.stdout.flush()        
		pickle.dump(self.in_mapping, open(dump_path + '_in_mapping.p', 'wb'))
		pickle.dump(self.out_mapping, open(dump_path + '_out_mapping.p', 'wb'))
		pickle.dump(self.edge_stat, open(dump_path + '_edge_stat.p', 'wb'))
		del self.in_mapping, self.out_mapping, self.edge_stat
		gc.collect()
		
		self.encoder['sum'] = offset
		print(self.encoder)
		sys.stdout.flush()  
		print('start encode input')
		sys.stdout.flush()  
		for i,ie in enumerate(self.input):
			self.input[i][1] += self.encoder[self.arg['types'][self.arg['edge_types'][ie[0]][0]]]
		self.input = np.split(self.input, [int(len(self.input)/split)*i for i in range(1,split)], axis=0)
		for index, each in enumerate(self.input):
			pickle.dump(each, open(dump_path + '_input.p.' + str(index), 'wb'))
		print('finish dump input')
		sys.stdout.flush()  
		del self.input            
		gc.collect()
		
		print('start encode output')
		sys.stdout.flush()  
		for i,ie in enumerate(self.output):
			self.output[i][1] += self.encoder[self.arg['types'][ie[0]]]
		self.output = np.split(self.output, [int(len(self.output)/split)*i for i in range(1,split)], axis=0)
		for index, each in enumerate(self.output):
			pickle.dump(each, open(dump_path + '_output.p.' + str(index), 'wb'))
		print('finish dump output')
		sys.stdout.flush()  
		pickle.dump(self.encoder, open(dump_path + '_offset.p', 'wb'))
        

# 	def dump(self, dump_path):
# 		print(self.edge_stat)
# 		cPickle.dump(self.encoder, open(dump_path + '_offset.p', 'wb'))
# 		cPickle.dump(self.input, open(dump_path + '_input.p', 'wb'))
# 		cPickle.dump(self.output, open(dump_path + '_output.p', 'wb'))
# 		cPickle.dump(self.in_mapping, open(dump_path + '_in_mapping.p', 'wb'))
# 		cPickle.dump(self.out_mapping, open(dump_path + '_out_mapping.p', 'wb'))
# 		cPickle.dump(self.edge_stat, open(dump_path + '_edge_stat.p', 'wb'))