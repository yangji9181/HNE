"""
Compute jaccard-like correlation, weighted and unweighted, between each pair of edge types
"""

import argparse
from collections import defaultdict
from numpy import mean, sqrt
from random import random

parser = argparse.ArgumentParser(description="Read in input and output filenames.")
parser.add_argument("--input", nargs="?", help="Input HIN filename.", type=str)
parser.add_argument("--output", nargs="?", help="Output correlation filename.", type=str)
parser.add_argument("--sample-rate", default=1.0, nargs="?", help="Sample rate on the center nodes.", type=float)
args = parser.parse_args()

"""
First pass on input file to find normalization factor for each edge type
"""
total_weights_dict = defaultdict(float)
normalization_multipliers_dict = {}
with open(args.input, "r") as f_in:
    for line in f_in:
        left_node, left_type, right_node, right_type, weight_str, edge_type = line.strip().split()
        total_weights_dict[edge_type] += float(weight_str)

    for edge_type in total_weights_dict:
        normalization_multipliers_dict[edge_type] = 1./total_weights_dict[edge_type]

"""
Second pass on input file to find egde type set for each center node type
"""
edge_type_set_per_node_type_dict = defaultdict(set)
node_set_per_node_type_dict = defaultdict(set)
node_edge_dict = defaultdict(dict)  # {center_node: {edge_type: {other_node: weight}}}
other_node_type_per_edge_type_dict = defaultdict(dict) # {edge_type: {center_node_type: other_node_type}}
with open(args.input, "r") as f_in:
    for idx, line in enumerate(f_in):
        left_node, left_type, right_node, right_type, weight_str, edge_type = line.strip().split()

#         left_node_split = left_node.split(":")
#         assert len(left_node_split) == 2, "Nodes must be given in the form [node-type]:[node-name], here we encounter: " + left_node
#         right_node_split = right_node.split(":")
#         assert len(right_node_split) == 2, "Nodes must be given in the form [node-type]:[node-name], here we encounter: " + right_node
#         left_node_type = left_node_split[0]
#         right_node_type = right_node_split[0]

        edge_type_set_per_node_type_dict[left_type].add(edge_type)
        edge_type_set_per_node_type_dict[right_type].add(edge_type)

        node_set_per_node_type_dict[left_type].add(left_node)
        node_set_per_node_type_dict[right_type].add(right_node)

        if edge_type not in node_edge_dict[right_node]:
            node_edge_dict[right_node][edge_type] = defaultdict(float)
        if edge_type not in node_edge_dict[left_node]:
            node_edge_dict[left_node][edge_type] = defaultdict(float)

        node_edge_dict[right_node][edge_type][left_node] += float(weight_str) * normalization_multipliers_dict[edge_type]
        node_edge_dict[left_node][edge_type][right_node] += float(weight_str) * normalization_multipliers_dict[edge_type]

        other_node_type_per_edge_type_dict[edge_type][left_type] = right_type
        other_node_type_per_edge_type_dict[edge_type][right_type] = left_type

        if idx % 5000 == 0:
            print("Line %d processed." % idx)

"""
Compute measures
"""
with open(args.output, "w") as f_out:
    print("center_node_type,other_node_i_type,other_node_j_type,edge_type_i,edge_type_j,inconsistancy", file=f_out)
    for center_node_type in edge_type_set_per_node_type_dict:
        cur_edge_type_list = list(edge_type_set_per_node_type_dict[center_node_type])

        for i, edge_type_i in enumerate(cur_edge_type_list):
            for edge_type_j in cur_edge_type_list[i+1:]:
                num_center_node = len(node_set_per_node_type_dict[center_node_type])
                num_center_node_processed = 0
                inverse_jac_list = []
                for center_node in node_set_per_node_type_dict[center_node_type]:
                    if random() > args.sample_rate:
                        continue

                    path_count_i_dict = defaultdict(float)
                    path_count_j_dict = defaultdict(float)

                    if (edge_type_i not in node_edge_dict[center_node]) or (edge_type_j not in node_edge_dict[center_node]):
                        continue

                    for other_node_i in node_edge_dict[center_node][edge_type_i]:
                        cur_weight = node_edge_dict[center_node][edge_type_i][other_node_i]
                        for linked_center_node in node_edge_dict[other_node_i][edge_type_i]:
                            if linked_center_node == center_node:  # do not consider itself
                                continue
                            path_count_i_dict[linked_center_node] += node_edge_dict[other_node_i][edge_type_i][linked_center_node] * cur_weight

                    for other_node_j in node_edge_dict[center_node][edge_type_j]:
                        cur_weight = node_edge_dict[center_node][edge_type_j][other_node_j]
                        for linked_center_node in node_edge_dict[other_node_j][edge_type_j]:
                            if linked_center_node == center_node:  # do not consider itself
                                continue
                            path_count_j_dict[linked_center_node] += node_edge_dict[other_node_j][edge_type_j][linked_center_node] * cur_weight

                    linked_center_node_union_set = set(path_count_i_dict) | set(path_count_j_dict)
                    if len(linked_center_node_union_set) == 0:
                        continue

                    numerator = 0.
                    denominator = 0.
                    for linked_center_node in linked_center_node_union_set:
                        cur_path_count_i = path_count_i_dict[linked_center_node]
                        cur_path_count_j = path_count_j_dict[linked_center_node]
                        numerator += min(cur_path_count_i, cur_path_count_j)
                        denominator += max(cur_path_count_i, cur_path_count_j)

                    if numerator > 0.:
                        inverse_jac_list.append((1.*denominator)/numerator)

                    num_center_node_processed += 1
                    if num_center_node_processed % 1000 == 0:
                        print("%d out of approximately %d * %f center nodes processed for center node type %s and and edge type pair %s and %s" % (num_center_node_processed, num_center_node, args.sample_rate, center_node_type, edge_type_i, edge_type_j))

                gamma = mean(inverse_jac_list) - 1 if inverse_jac_list else 0.  # inconsistancy score

                print("%s,%s,%s,%s,%s,%.6g" % tuple([center_node_type, other_node_type_per_edge_type_dict[edge_type_i][center_node_type], other_node_type_per_edge_type_dict[edge_type_j][center_node_type], edge_type_i, edge_type_j, gamma]), file=f_out)
