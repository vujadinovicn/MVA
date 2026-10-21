"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    ##################
    # your code here #
    ##################
    # compute the laplacian matrix L
    A = nx.adjacency_matrix(G)
    inv_D = diags([1/deg for _, deg in G.degree()], format='csr') # create diagonal degree matrix; use csr for better efficiency
    I = eye(G.number_of_nodes(), format='csr') # use csr for better efficiency
    L = I - inv_D @ A

    # compute the k smallest eigenvectors of L
    _, eigenvectors = eigs(L, k, which='SR')
    U = np.real(eigenvectors)

    # compute k-means on rows of U to find clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(U)
    clustering = {node: clusters[i] for i, node in enumerate(G.nodes())}
    
    return clustering


############## Task 4

##################
# your code here #
##################
G = nx.read_edgelist("datasets/CA-HepTH.txt", comments="#", delimiter="\t")
largest_cc = max(nx.connected_components(G), key=len)
G_largest_cc = G.subgraph(largest_cc).copy()
spec_clustering = spectral_clustering(G_largest_cc, k=50)

# for value in sorted(set(spec_clustering.values())):
#     print(f"Cluster {value}: {len([_ for _, cluster in spec_clustering.items() if cluster == value])}")

############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    modularity = 0

    m = G.number_of_edges()
    communities = list(set(clustering.values())) # get unique community labels

    for community in communities:
        community_nodes = [node for node, cluster in clustering.items() if cluster == community] # find all nodes in the community
        community_graph = G.subgraph(community_nodes) # create subgraph for the community
        
        lc = community_graph.number_of_edges() # no. edges inside the community
        d = sum([G.degree(node) for node in community_nodes]) # sum of degrees of the nodes that belong to community

        modularity += lc / m - (d / (2 * m)) ** 2
    ##################
    
    return modularity


############## Task 6

##################
# your code here #
##################
modularity_spec_clustering = modularity(G_largest_cc, spec_clustering)
print(f"Modularity obtained by spectral clustering with 50 clusters is {np.round(modularity_spec_clustering, 6)}.")

random_clustering = {node: randint(0,49) for node in G_largest_cc.nodes()}
modularity_random_clustering = modularity(G_largest_cc, random_clustering)
print(f"Modularity obtained by random clustering with 50 clusters is {np.round(modularity_random_clustering, 6)}.")






