import sys
from itertools import chain, combinations
import operator
import networkx as nx

def find_powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return map(set, chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1)))

def is_connected(cand_aspect):  # ((node_type_1, node_type_2, edge_type), ...)
    subgraph = nx.Graph()
    for node_type_1, node_type_2, edge_type in cand_aspect:
        subgraph.add_edge(node_type_1, node_type_2)
    return nx.is_connected(subgraph)

input_file = sys.argv[1]  # python calc_aspect_inconsistency.py XXX_corr.csv
output_file = sys.argv[2]

atom_score_dict = {}
edge_type_set = set()
with open(input_file, "r") as f_in:
    first_line_split = f_in.readline().strip().split(",")
    for line in f_in:
        center_node_type, other_node_i_type, other_node_j_type, edge_type_i, edge_type_j, gamma_str = line.strip().split(",")
        gamma = float(gamma_str)
        edge_type_set.add((center_node_type, other_node_i_type, edge_type_i))
        edge_type_set.add((center_node_type, other_node_j_type, edge_type_j))
        atom_score_dict[(edge_type_i, edge_type_j)] = gamma

aspect_score_dict = {}
for cand_aspect in find_powerset(edge_type_set):  # currently implemented with brutal force
    if len(cand_aspect) < 2:
        continue
    if not is_connected(cand_aspect):
        continue
    cur_aspect_score = 0.
    for aug_edge_type_pair in combinations(cand_aspect, 2):
        edge_type_pair = (aug_edge_type_pair[0][2], aug_edge_type_pair[1][2])
        if edge_type_pair not in atom_score_dict:
            edge_type_pair = (edge_type_pair[1], edge_type_pair[0])
        if edge_type_pair not in atom_score_dict:
            continue
        cur_aspect_score += atom_score_dict[edge_type_pair]

    aspect_score_dict[tuple(cand_aspect)] = cur_aspect_score

aspect_score_sorted_list = sorted(aspect_score_dict.items(), key=operator.itemgetter(1))

with open(output_file, "w") as f_out:
    for aspect_socre in aspect_score_sorted_list:
        cur_node_set = set()
        cur_edge_list = set()
        for aug_edge in aspect_socre[0]:
            cur_node_set.add(aug_edge[0])
            cur_node_set.add(aug_edge[1])
            cur_edge_list.add(aug_edge[2])
        print("Nodes", cur_node_set, "Edges", cur_edge_list, file=f_out)
        print("Inc", aspect_socre[1], file=f_out)


