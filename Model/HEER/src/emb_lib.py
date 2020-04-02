import sys
import neg
import torch.optim as optim
import torch as t
import numpy as np
import torch.utils.data as tdata
import utils
import math
import time
import pickle
import numpy as np
# from tqdm import tqdm

# support heterogenous node type
# first consider about first order proximity
# different node type and edge type mapping function

class SkipGram(object):
	"""pytorch implementation for SkipGram"""
	def __init__(self, arg):
		super(SkipGram, self).__init__()
		type_offset = pickle.load(open(arg['data_dir'] + arg['graph_name'] + '_offset.p'))
		self.neg_loss = neg.NEG_loss(type_offset=type_offset, node_types=arg['node_types'], edge_types=arg['edge_types'], 
			embed_size=arg['emb_size'], pre_train_path=arg['pre_train'], 
			graph_name=arg['graph_name'], mode=arg['mode'], map_mode=arg['map_mode'])
		self.input = arg['network']
		edge_stats = pickle.load(open(arg['data_dir'] + arg['graph_name'] + '_edge_stat.p'))
		self.mode = arg['mode']
		self.map_mode = arg['map_mode']
		self.dump_timer = arg['dump_timer']
		self.model_dir = arg['model_dir']
		self.log_dir = arg['log_dir']
		self.fine_tune = arg['fine_tune']
		self.lr = arg['lr']
		self._params = []
		if self.map_mode != -1:
			self._params = [{'params': self.neg_loss.in_embed.parameters()}, 
				{'params': self.neg_loss.out_embed.parameters()}]
			for i in xrange(len(self.neg_loss.edge_mapping)):
				self._params.append({'params': self.neg_loss.edge_mapping[i].parameters(),
					'lr': arg['lr'] * arg['lr_ratio'] * (float(len(self.input))) / (type_offset['sum'] * edge_stats[i] + 1e-6)})
		
		self.window_size = arg['window_size']
		self.graph_name = arg['graph_name']
		
		self.data = tdata.DataLoader(self.input, arg['batch_size'], shuffle=True)
		self.batch_size = arg['batch_size']
		self.iter = arg['iter']
		self.neg_ratio = arg['neg_ratio']
	
	# support fine tune 
	def freeze_embedding(self):
		self.SGD = optim.SGD(self.neg_loss.parameters(), lr = self.lr)
		for param in self.neg_loss.in_embed.parameters():
			param.requires_grad = False
		for param in self.neg_loss.out_embed.parameters():
			param.requires_grad = False
		
	# support fine tune
	def update_embedding(self):
		for param in self.neg_loss.in_embed.parameters():
			param.requires_grad = True
		for param in self.neg_loss.out_embed.parameters():
			param.requires_grad = True
		if self.map_mode != -1:
			self.SGD = optim.SGD(self._params, lr = self.lr)
		else:
			self.SGD = optim.SGD(self.neg_loss.parameters(), lr = self.lr)

	def train(self):
		print("Number of batches " + str(len(self.data)))
		sys.stdout.flush()
		report = int(len(self.data)/10)
		self.neg_loss.train()
		self.freeze_embedding()
		with open(self.log_dir + 'heer_' + self.graph_name + '_op_' + str(self.mode) + 
						'_mode_' + str(self.map_mode)+ '.log', 'w') as LOG:
			for epoch in xrange(self.iter):
				print('Start epoch ' + str(epoch) + " at " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
				sys.stdout.flush()
				loss_sum = 0
				if epoch == self.fine_tune:
					self.update_embedding()
					if self.map_mode > 1:
						print("finish fine tuning for deep edge archs")
						sys.stdout.flush()
				for i, data in enumerate(self.data, 0):
					if i%report==0: 
						print("Start batch " + str(i) + " at " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()))
						sys.stdout.flush()
					inputs, labels = data
					loss, pure_loss = self.neg_loss(inputs, labels, self.neg_ratio)
					
					if np.isnan(loss.data.cpu().numpy()):
						return -1
					loss_sum += pure_loss * self.batch_size
					self.SGD.zero_grad()
					
					loss.backward()
					
					# utils.clip_sparse_grad_norm(self.neg_loss.in_embed.parameters(), 0.1)
					# utils.clip_sparse_grad_norm(self.neg_loss.out_embed.parameters(), 0.1)
					
					#for i in xrange(len(self.neg_loss.edge_mapping)):
					#	utils.clip_grad_norm(self.neg_loss.edge_mapping[i].parameters(), 0.1)

					self.SGD.step()
					

				if epoch % self.dump_timer == 0:
					model_path = self.model_dir + 'heer_' + self.graph_name + '_' + str(epoch) + '_op_' + str(self.mode) + '_mode_' + str(self.map_mode)+ '.pt'
					t.save(self.neg_loss.state_dict(), model_path)

				LOG.write(str(epoch) + '\t' + str(np.asscalar(loss_sum.data.cpu().numpy())) + '\n')
				LOG.flush()


	def output(self):
        
		input_emb = self.neg_loss.input_embeddings()
		output_emb = self.neg_loss.output_embeddings()
		
		return np.hstack([output_emb, input_emb])
		