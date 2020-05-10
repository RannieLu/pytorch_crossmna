# encoding:utf8
import numpy as np
import random
def AUC(node2vec, nodes, edges):
    """
    :param node2vec: the node embedding of given layer
    :param test_graph: test set
    :param i: i-th layer
    :return:
    """
    a = 0
    b = 0
    for edge in edges:
        i = edge[0]
        j = edge[1]
        if i in node2vec and j in node2vec:
            dot1 = np.dot(node2vec[i], node2vec[j])
            random_node = random.sample(nodes, 1)[0]
            while random_node == j or (not (random_node in node2vec)):
                random_node = random.sample(nodes, 1)[0]  #if the random_node ==i or == the other node has edge with i ?
            dot2 = np.dot(node2vec[i], node2vec[random_node])
            if dot1 > dot2:
                a += 1
            elif dot1 == dot2:
                a += 0.5
            b += 1
    if b == 0:
        return False
    print("a ", a, "b ", b)
    return float(a) / b


def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]

