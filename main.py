from AUC import *
from train_crossmna import train_crossmna
import numpy as np
from lis import *
from utils import *




file_name = "data/CKM-Physicians-Innovation_multiplex.edges"
#file_name = "data/Vickers-Chan-7thGraders_multiplex.edges"
edge_data_by_type, _, all_nodes = load_network_data(file_name)

number_of_groups = 5
edge_data_by_type_by_group = dict()
for edge_type in edge_data_by_type.keys():
    # print("edge_type ", type(edge_type))
    all_data = edge_data_by_type[edge_type]
    separated_data = divide_data(all_data, number_of_groups)
    edge_data_by_type_by_group[edge_type] = separated_data


for i in range(number_of_groups):
    training_data_by_type = dict()
    evaluation_data_by_type = dict()
    for edge_type in edge_data_by_type_by_group:
        training_data_by_type[edge_type] = list()
        evaluation_data_by_type[edge_type] = list()
        for j in range(number_of_groups):
            if j == i:
                for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                    evaluation_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))
            else:
                for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                    #print('node_type', type(tmp_edge[0]))
                    training_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1]))

    base_edges = list()
    training_nodes = list()
    for edge_type in training_data_by_type:
        for edge in training_data_by_type[edge_type]:
            base_edges.append(edge)
            training_nodes.append(edge[0])
            training_nodes.append(edge[1])
    training_nodes = list(set(training_nodes))
    training_data_by_type['Base'] = base_edges
    # select  the false and true edge
        # liscne
    training_data_no_base = dict(training_data_by_type)
    del training_data_no_base['Base']
    crossmna_model = train_crossmna(training_data_no_base,undirected=False)
    for edge_type in training_data_by_type.keys():
        if edge_type == 'Base':
            continue
        selected_true_edges = list()
        tmp_training_nodes = list()
        for edge in training_data_by_type[edge_type]:
            tmp_training_nodes.append(edge[0])
           # tmp_training_nodes.append(edge[1])
        tmp_training_nodes = list(set(tmp_training_nodes))
        print(len(tmp_training_nodes))
        temp_crossmna_score = 0
        if edge_type != "Base":
            crossmna_dict = dict()
            crossmna_type = crossmna_model[int(edge_type)]
            for node in tmp_training_nodes:
                crossmna_dict[node] = crossmna_type[int(node)]
            temp_crossmna_score += AUC(crossmna_dict, tmp_training_nodes,edge_data_by_type[edge_type])
            print("temp_crossmna_score", temp_crossmna_score)

        
