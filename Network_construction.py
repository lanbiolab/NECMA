import numpy as np
import data_preprocessing
def constructNetwork(circRNA_miRNA_network):
    circRNA_similarity = np.mat(np.zeros((circRNA_miRNA_network.shape[0], circRNA_miRNA_network.shape[0])))
    miRNA_similarity = np.mat(np.zeros((circRNA_miRNA_network.shape[1], circRNA_miRNA_network.shape[1])))

    circNetwork = data_preprocessing.circRNA_GIP(circRNA_miRNA_network, circRNA_similarity)
    miNetwork = data_preprocessing.miRNA_GIP(circRNA_miRNA_network, miRNA_similarity)

    mat1 = np.hstack((circNetwork, circRNA_miRNA_network))
    mat2 = np.hstack((circRNA_miRNA_network.T, miNetwork))
    finalNetwork = np.vstack((mat1, mat2))

    return finalNetwork