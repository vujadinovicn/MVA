"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():

    ##################
    # your code here #
    ##################

    dataset = TUDataset('datasets', name='MUTAG')
    Gs = [to_networkx(data, to_undirected=True) for data in dataset] # transform graphs to networkx
    y = [data.y.item() for data in dataset] # get labels
    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(Gs_train), 4))
    
    ##################
    # your code here #
    ##################

    np.random.seed(0)
    # compute phi_train
    for i in range(len(Gs_train)): # iterate over all train graphs
        curr_G = Gs_train[i]
        nodes = list(curr_G.nodes())
        for _ in range(n_samples): # sample n_samples times
            sampled_nodes = np.random.choice(nodes, size=3, replace=False)# sample 3 nodes from current (w/o replacement)
            subgraph = curr_G.subgraph(sampled_nodes)
            # if subgraph is isomorphic to one of the graphlets, increment phi_train and go to next sample
            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    phi_train[i,j] += 1
                    break

    phi_test = np.zeros((len(Gs_test), 4))
    
    ##################
    # your code here #
    ##################

    # compute phi_test
    for i in range(len(Gs_test)): # iterate over all test graphs
        curr_G = Gs_test[i]
        nodes = list(curr_G.nodes())
        for _ in range(n_samples): # sample n_samples times
            sampled_nodes = np.random.choice(nodes, size=3, replace=False) # sample 3 nodes from current (w/o replacement)
            subgraph = curr_G.subgraph(sampled_nodes)
            # if subgraph is isomorphic to one of the graphlets, increment phi_test and go to next sample
            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    phi_test[i,j] += 1
                    break

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9

##################
# your code here #
##################
K_train_gk, K_test_gk = graphlet_kernel(G_train, G_test) # compute kernel matrices using graphlet kernel


############## Task 10

##################
# your code here #
##################
# compare SVM accuracy for both kernels
clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)
y_pred_sp = clf_sp.predict(K_test_sp)
print(f"Accuracy of SVM with shortest path is {accuracy_score(y_test, y_pred_sp)}")

clf_gk = SVC(kernel='precomputed')
clf_gk.fit(K_train_gk, y_train)
y_pred_gk = clf_gk.predict(K_test_gk)
print(f"Accuracy of SVM with graphlet kernel is {accuracy_score(y_test, y_pred_gk)}")
