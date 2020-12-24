import numpy as np

#Calculate circRNA similarity
def circRNA_GIP(association_network, similarity_network):
    pard = 0
    for i in range(0,association_network.shape[0]):
        pard = pard + np.sum(np.multiply(association_network[i,],association_network[i,]))

    pard1 = association_network.shape[0] / pard

    for m in range(0,association_network.shape[0]):
        for n in range(0,association_network.shape[0]):
            minus = association_network[m,] - association_network[n,]
            similarity_network[m,n] = np.exp(-pard1 * np.sum(np.multiply(minus,minus)))

    return similarity_network

#Calculate miRNA similarity
def miRNA_GIP(association_network, similarity_network):
    pard = 0
    for i in range(0,association_network.shape[1]):
        pard = pard + np.sum(np.multiply(association_network[:,i],association_network[:,i]))

    pard1 = association_network.shape[1] / pard

    for o in range(0,association_network.shape[1]):
        for p in range(0,association_network.shape[1]):
            minus = association_network[:,o] - association_network[:,p]
            similarity_network[o,p] = np.exp(-pard1 * np.sum(np.multiply(minus,minus)))

    return similarity_network

