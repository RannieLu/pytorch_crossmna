import networkx as nx
from gensim.models import Word2Vec

""" load data"""
def load_network_data(fname):
    print("loading data from ", fname)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(fname, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            all_nodes.append(words[2])
            all_nodes.append(words[1])
            all_edges.append((words[1], words[2]))
            if words[0] not in edge_data_by_type.keys():
                edge_data_by_type[words[0]] = list()
            edge_data_by_type[words[0]].append((words[1], words[2]))
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    #create a common layer
    edge_data_by_type['Base'] = all_edges
    print("finish loading data")
    return edge_data_by_type, all_edges, all_nodes

#build the network
def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G

    
"""train common embedding"""

def  train_common_model(walks, vector_size = 100,iteration=None):
    
    if iteration == None:
        iteration = 100
    
    model = Word2Vec(walks, size=vector_size, window=5, min_count=0, sg=1, workers=4, iter=iteration)
    return model


def train_deepwalk_embedding(walks, iteration=None):
    if iteration is None:
        iteration = 100
    model = Word2Vec(walks, size=200, window=5, min_count=0, sg=1, workers=4, iter=iteration)
    return model
    
    


