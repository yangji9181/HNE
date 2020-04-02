import sys
import argparse
import numpy as np
import os
from emb_lib import SkipGram
import network as nx
import torch as t
import pickle
import utils
import torch.utils.data as tdata

global config

def parse_args():

	parser = argparse.ArgumentParser(description="Run heer.")
	parser.add_argument('--build-graph', type=int, help='heterogeneous information network construction') 
    
	parser.add_argument('--link', nargs='?', help='Input graph path')
	parser.add_argument('--config', nargs='?', help='Input config path')
	parser.add_argument('--temp-dir', type=str, default='', help='temp data directory')
	parser.add_argument('--data-name', type=str, default='', help='prefix of dumped data')    
    
	parser.add_argument('--pre-train-path', type=str, default='', help='embedding initialization')
	parser.add_argument('--output', type=str, default='', help='embedding output')
	parser.add_argument('--model-dir', type=str, default='', help='model directory')
	parser.add_argument('--log-dir', type=str, default='', help='log directory')    
	parser.add_argument('--dimensions', type=int, default=50, help='Number of dimensions. Default is 50.')
	parser.add_argument('--gpu', nargs='?', default='0', help='Embeddings path')
	parser.add_argument('--rescale', type=float, default=0.1) 
	parser.add_argument('--lr', type=int, default=10) 
	parser.add_argument('--lrr', type=int, default=10)
	parser.add_argument('--batch-size', type=int, default=50, help='Batch size. Default is 50.') 
	parser.add_argument('--iter', default=50, type=int, help='Number of epochs in SGD')
	parser.add_argument('--dump-timer', default=100, type=int)
	parser.add_argument('--op', default=1, type=int)
	parser.add_argument('--map-func', default=0, type=int)
	parser.add_argument('--fine-tune', type=int, default=0, help='fine tune phase')

	return parser.parse_args()


def learn_embeddings(split=10):

	_data = args.rescale * utils.load_emb(args.temp_dir, args.data_name, args.pre_train_path, int(args.dimensions/2), config['nodes'])
	_network = tdata.TensorDataset(
        t.LongTensor(np.vstack([pickle.load(open(args.temp_dir + args.data_name + '_input.p.' + str(i))) for i in range(split)])), 
        t.LongTensor(np.vstack([pickle.load(open(args.temp_dir + args.data_name + '_output.p.' + str(i))) for i in range(split)])))
    
	model = SkipGram({'emb_size':int(args.dimensions/2),
		'window_size':1, 'batch_size':args.batch_size, 'iter':args.iter, 'neg_ratio':5, 
		'lr_ratio':args.lrr, 'lr': args.lr, 'network':_network, 
		'pre_train':_data, 'node_types':config['nodes'], 'edge_types':config['edges'],
		'graph_name':args.data_name, 'dump_timer':args.dump_timer, 'data_dir':args.temp_dir, 
		'mode':args.op, 'map_mode':args.map_func, 'fine_tune':args.fine_tune, 'model_dir':args.model_dir, 'log_dir':args.log_dir})

	model.train()
    
	return model.output()

def output(args, config, embs):
	
	offset,prev_offset = 0,0
	type_offset = pickle.load(open(args.temp_dir + args.data_name + '_offset.p')) 
	out_mapping = pickle.load(open(args.temp_dir + args.data_name + '_out_mapping.p'))

	with open(args.output, 'w') as OUT:
		OUT.write('size={}, iter={}, batch_size={}, rescale={}, lr={}, lr_ratio={}\n'.format(args.dimensions, args.iter, args.batch_size, args.rescale, args.lr, args.lrr))
		config['nodes'].append('sum')        
		for idx,t in enumerate(config['nodes']):
			if t=='sum': break
			tp = config['nodes'][idx+1]
			while offset < type_offset[tp]:
				OUT.write("{}\t{}\n".format(out_mapping[t][offset-prev_offset], ' '.join(map(str, embs[offset].tolist()))))
				offset += 1
			prev_offset = type_offset[tp]
		    
        

def main(args):
    
	global config
	config = utils.read_config(args.config)
	if args.build_graph:
		print('Build Graph')
		sys.stdout.flush()
		tmp = nx.HinLoader({'graph': args.link, 'types':config['nodes'], 'edge_types':config['edges']})
		print('start readHin')
		sys.stdout.flush()
		tmp.readHin(config['types'])
		print('start encode')
		sys.stdout.flush()
		tmp.encode(args.temp_dir+args.data_name)
	else:
		print('Learn Embeddings')
		sys.stdout.flush()
		embs = learn_embeddings()
		output(args, config, embs)
        

if __name__ == "__main__":
	args = parse_args()
	t.cuda.set_device(int(args.gpu))
	main(args)