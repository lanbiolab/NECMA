import numpy as np

def restructuring_matrix (circ_embedding, mi_embedding):
    alpa = 0.6
    first_network = np.dot(alpa*circ_embedding, (1-alpa)*mi_embedding.T)

    inner_network = np.dot(circ_embedding, mi_embedding.T)

    rows = first_network.shape[0]
    cols = first_network.shape[1]
    mid1_network = np.mat(np.zeros((rows, cols)))
    mid2_network = np.mat(np.zeros((rows, cols)))
    score_network = np.mat(np.zeros((rows, cols)))
    final_network = np.mat(np.zeros((rows, cols)))
    for i in range(0, rows):
        for j in range(0, cols):
            mid1_network[i, j] = np.exp(first_network[i, j])

    for i in range(0, rows):
        for j in range(0, cols):
            mid2_network[i, j] = 1 + np.exp(first_network[i, j])

    # predicted circRNA-miRNA association
    for i in range(0, rows):
        for j in range(0, cols):
            final_network[i, j] = max(score_network[i, j], inner_network[i, j])

    return final_network


