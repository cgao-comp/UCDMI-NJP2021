import scipy.sparse as sp

def getJaccard_similarity(node_num, graph):
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    node_list = list(graph.node())
    for i, node in enumerate(node_list):
        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list
        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            fenzi = len(list(set(first_neighbor).intersection(set(neibor_j_list))))
            fenmu = len(list(set(first_neighbor).union(set(neibor_j_list))))
            similar_matrix[node, node_j] = fenzi / fenmu
    return similar_matrix