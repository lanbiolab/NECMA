import numpy as np
import pandas as pd
import Network_construction
import matrix_decomposition
import Netmf

#The purpose of this demo is to show the calculation process of the NECMA model.
csv_data = pd.read_csv('circRNA-microRNA.csv',index_col= False,header=None)

init_network = np.mat(csv_data)
init_network = np.delete(init_network, 0, axis=0)
init_network = np.delete(init_network, 0, axis=1)
init_network = init_network.astype(np.float)

circRNA_miRNA_net = Network_construction.constructNetwork(init_network)
circRNA_miRNA_embeding = Netmf.netmf_small(circRNA_miRNA_net, 1, 1, 8)
miRNA_length = init_network.shape[1]
miRNA_embeding = np.array(circRNA_miRNA_embeding[0:miRNA_length, :])
circRNA_embeding = np.array(circRNA_miRNA_embeding[miRNA_length::, :])

score_matrix = matrix_decomposition.restructuring_matrix(circRNA_embeding, miRNA_embeding)

