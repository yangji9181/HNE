import numpy as np
from collections import defaultdict


data_folder, model_folder = '../Data', '../Model'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'


def metapath2vec_esim_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/metapath2vec-ESim/data/{dataset}'
    
    print(f'metapath2vec-ESim: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}','w')
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]} {line[2]}\n')    
    new_node_file.close()
    
    print(f'metapath2vec-ESim: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, _, _ = line[:-1].split('\t')
            new_link_file.write(f'{left} {right}\n')
    new_link_file.close()
    
    print(f'metapath2vec-ESim: writing {dataset}\'s path file!')
    next_node = defaultdict(list)
    with open(f'{ori_data_folder}/{info_file}','r') as original_info_file:
        start = False
        for line in original_info_file:
            if line[:4]=='LINK': 
                start=True
                continue
            if start and line[0]=='\n':
                break
            if start:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x)!=0, line))
                next_node[snode].append(enode)
    with open(f'{model_data_folder}/path.dat','w') as new_path_file:
        for start, ends in next_node.items():
            for end in ends:
                new_path_file.write(f'{start}{end} 1.0\n')
                if end in next_node:
                    for twohop in next_node[end]:
                        new_path_file.write(f'{start}{end}{twohop} 0.5\n')
       
    return


def pte_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/PTE/data/{dataset}'
    
    print(f'PTE: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}','w')
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]}\n')    
    new_node_file.close()
    
    print(f'PTE: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {right} {ltype} {weight}\n')
    new_link_file.close()
    
    print(f'PTE: writing {dataset}\'s type file!')    
    type_count = 0
    with open(f'{ori_data_folder}/{meta_file}','r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, _ = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity=='Edge' and info[0]=='Type':
                type_count += 1
    new_type_file = open(f'{model_data_folder}/type.dat','w')
    new_type_file.write(f'{type_count}\n')
    new_type_file.close()
        
    return


def hin2vec_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HIN2Vec/data/{dataset}'
    
    print(f'HIN2Vec: reading {dataset}\'s node file!')    
    type_dict = {}
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            type_dict[line[0]] = line[2]
    
    print(f'HIN2Vec: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{type_dict[left]}\t{right}\t{type_dict[right]}\t{ltype}\n')
    new_link_file.close()
    
    return


def aspem_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/AspEm/data/{dataset}'   
            
    print(f'AspEm: converting {dataset}\'s node file!')
    type_dict = {}
    new_node_file = open(f'{model_data_folder}/{node_file}','w')
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            type_dict[line[0]] = line[2]
            new_node_file.write(f'{line[2]}:{line[0]} {line[2]}\n')    
    new_node_file.close()
    
    print(f'AspEm: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{type_dict[left]}:{left} {type_dict[left]} {type_dict[right]}:{right} {type_dict[right]} {weight} {ltype}\n')
    new_link_file.close()
    
    print(f'AspEm: writing {dataset}\'s type file!')  
    type_count, target_type = 0, -1
    with open(f'{ori_data_folder}/{meta_file}','r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, _ = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity=='Node' and info[0]=='Type':
                type_count += 1
            if entity=='Label' and info[0]=='Class':
                target_type = info[1]
                break
    new_type_file = open(f'{model_data_folder}/type.dat','w')
    new_type_file.write(f'{target_type}\n')
    new_type_file.write(f'{type_count}\n')
    new_type_file.close()
    
    return


def heer_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HEER/data/{dataset}'
    
    print(f'HEER: reading {dataset}\'s node file!')    
    type_dict, types = {}, set()
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            type_dict[line[0]] = line[2]
            types.add(line[2])
            
    print(f'HEER: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{type_dict[left]}:{left} {type_dict[right]}:{right} {weight} {ltype}:d\n')
    new_link_file.close() 
    
    print(f'HEER: writing {dataset}\'s config file!')    
    edge_info = []
    with open(f'{ori_data_folder}/{info_file}','r') as original_info_file:
        start = False
        for line in original_info_file:
            if line[:4]=='LINK': 
                start=True
                continue
            if start and line[0]=='\n':
                break
            if start:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x)!=0, line))
                edge_info.append([ltype, snode, enode])
    edge_info = np.array(edge_info).astype(int)            
    with open(f'{model_data_folder}/config.dat','w') as new_config_file:
        new_config_file.write(f'{edge_info[:,1:].tolist()}\n')
        new_config_file.write(f'{np.arange(len(types)).astype(str).tolist()}\n')
        temp = list(map(lambda x: f'{x}:d', edge_info[:,0].tolist()))
        new_config_file.write(f'{temp}\n')
        new_config_file.write(f'{np.ones(len(edge_info)).astype(int).tolist()}\n')
        
    return


def rgcn_convert(dataset, attributed, supervised):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/R-GCN/data/{dataset}'
    
    entity_count, relation_count = 0, 0
    with open(f'{ori_data_folder}/{meta_file}','r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, count = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity=='Node' and info[0]=='Total': entity_count = int(count)
            elif entity=='Edge' and info[0]=='Type': relation_count += 1
                        
    print(f'R-GCN: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w') 
    new_link_file.write(f'{entity_count} {relation_count}\n')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {ltype} {right}\n')
    new_link_file.close() 
    
    if attributed=='True':
        print(f'R-GCN: converting {dataset}\'s node file for attributed training!')
        new_node_file = open(f'{model_data_folder}/{node_file}','w')
        with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
            for line in original_node_file:
                line = line[:-1].split('\t')
                new_node_file.write(f'{line[0]}\t{line[3]}\n')
        new_node_file.close()

    if supervised=='True':
        print(f'R-GCN: converting {dataset}\'s label file for semi-supervised training!')
        new_label_file = open(f'{model_data_folder}/{label_file}','w')
        with open(f'{ori_data_folder}/{label_file}','r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')    
        new_label_file.close()
    
    return


def han_convert(dataset, attributed, supervised):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HAN/data/{dataset}'   
            
    print('HAN: converting {}\'s node file for {} training!'.format(dataset, 'attributed' if attributed=='True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}','w')
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            if attributed=='True': new_node_file.write(f'{line[0]}\t{line[2]}\t{line[3]}\n')
            elif attributed=='False': new_node_file.write(f'{line[0]}\t{line[2]}\n')
    new_node_file.close()
    
    print(f'HAN: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
    new_link_file.close()
    
    print(f'HAN: writing {dataset}\'s config file!') 
    target_node, target_edge, ltypes = 0, 0, []
    with open(f'{ori_data_folder}/{info_file}','r') as original_info_file:
        for line in original_info_file:
            if line.startswith('Targeting: Link Type'): target_edge = int(line[:-2].split(',')[-1])
            if line.startswith('Targeting: Label Type'): target_node = int(line.split(' ')[-1])
    with open(f'{ori_data_folder}/{info_file}','r') as original_info_file: 
        lstart = False
        for line in original_info_file:
            if line.startswith('LINK'): 
                lstart=True
                continue
            if lstart and line[0]=='\n': break
            if lstart:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x)!=0, line))
                ltypes.append((snode, enode, ltype))
    config_file = open(f'{model_data_folder}/config.dat','w')
    config_file.write(f'{target_node}\n')
    config_file.write(f'{target_edge}\n')
    config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
    config_file.close()
    
    if supervised=='True':
        print(f'HAN: converting {dataset}\'s label file for semi-supervised training!')
        new_label_file = open(f'{model_data_folder}/{label_file}','w')
        with open(f'{ori_data_folder}/{label_file}','r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')  
        new_label_file.close()
    
    return


def hgt_convert(dataset, attributed, supervised):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/HGT/data/{dataset}'
    
    print('HGT: converting {}\'s node file for {} training!'.format(dataset, 'attributed' if attributed=='True' else 'unattributed'))
    new_node_file = open(f'{model_data_folder}/{node_file}','w')
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            if attributed=='True': new_node_file.write(f'{line[0]}\t{line[2]}\t{line[3]}\n')
            elif attributed=='False': new_node_file.write(f'{line[0]}\t{line[2]}\n')
    new_node_file.close()    
    
    print(f'HGT: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{right}\t{ltype}\n')
    new_link_file.close()
    
    if supervised=='True':
        print(f'HGT: converting {dataset}\'s label file for semi-supervised training!')
        labeled_type, nlabel, begin = None, -1, False
        with open(f'{ori_data_folder}/{info_file}', 'r') as file:
            for line in file:
                if line.startswith('Targeting: Label Type'): 
                    labeled_type = int(line.split(' ')[-1])
                elif line=='TYPE\tCLASS\tMEANING\n':
                    begin = True
                elif begin:
                    nlabel += 1    
        new_label_file = open(f'{model_data_folder}/{label_file}','w')
        new_label_file.write(f'{labeled_type}\t{nlabel}\n')
        with open(f'{ori_data_folder}/{label_file}','r') as original_label_file:
            for line in original_label_file:
                line = line[:-1].split('\t')
                new_label_file.write(f'{line[0]}\t{line[3]}\n')
        new_label_file.close()
        
    return


def transe_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/TransE/data/{dataset}'
    
    entity_count, relation_count, triplet_count = 0, 0, 0
    with open(f'{ori_data_folder}/{meta_file}','r') as original_meta_file:
        for line in original_meta_file:
            entity, info, _, count = line[:-1].split(' ')
            info = info[:-1].split('_')
            if entity=='Node' and info[0]=='Total': entity_count = int(count)
            elif entity=='Edge' and info[0]=='Total': triplet_count = int(count)
            elif entity=='Edge' and info[0]=='Type': relation_count += 1
    
    print(f'TransE: converting {dataset}\'s node file!')
    new_node_file = open(f'{model_data_folder}/{node_file}','w')
    new_node_file.write(f'{entity_count}\n')
    with open(f'{ori_data_folder}/{node_file}','r') as original_node_file:
        for line in original_node_file:
            line = line[:-1].split('\t')
            new_node_file.write(f'{line[0]} {line[0]}\n')    
    new_node_file.close()
    
    print(f'TransE: converting {dataset}\'s link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w') 
    new_link_file.write(f'{triplet_count}\n')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left} {right} {ltype}\n')
    new_link_file.close()    
    
    print(f'TransE: writing {dataset}\'s relation file!')
    with open(f'{model_data_folder}/rela.dat','w') as new_rela_file:
        new_rela_file.write(f'{relation_count}\n')
        for each in range(relation_count):
            new_rela_file.write(f'{each} {each}\n')     
    
    return


def distmult_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/DistMult/data/{dataset}'
                        
    print(f'DistMult: converting link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{ltype}\t{right}\n')
    new_link_file.close()         
    
    return


def conve_convert(dataset):
    
    ori_data_folder = f'{data_folder}/{dataset}'
    model_data_folder = f'{model_folder}/ConvE/data/{dataset}'
                        
    print(f'ConvE: converting link file!')
    new_link_file = open(f'{model_data_folder}/{link_file}','w')
    with open(f'{ori_data_folder}/{link_file}','r') as original_link_file:
        for line in original_link_file:
            left, right, ltype, weight = line[:-1].split('\t')
            new_link_file.write(f'{left}\t{ltype}\t{right}\n')
    new_link_file.close()         
    
    return