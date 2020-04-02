import network


def load_a_HIN(fname):
    
    g = load_a_HIN_from_edge_file(fname)
    return g


def load_a_HIN_from_edge_file(fname):
    '''
        Load a HIN from a file which contains edges of the HIN

        In the file, each line is an edge, formated as
        <source_node> <source_class> <dest_node> <dest_class> <edge_class>
        An example file: res/karate_club_edges.txt

        It is assumed that the HIN is directed
    '''
    g = network.HIN()
    with open(fname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            src, src_class, dst, dst_class, edge_class = line.split('\t')
            g.add_edge(src, src_class, dst, dst_class, edge_class)
    g.print_statistics()
    return g

