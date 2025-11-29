"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx

############## Task 1

G = nx.read_edgelist("datasets/CA-HepTH.txt", comments="#", delimiter="\t")
print(f"There are {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in the graph.")


# ############## Task 2

print(f"There are {nx.number_connected_components(G)} connected components in the graph.")

if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest_cc = G.subgraph(largest_cc)
    print(f"There are {G_largest_cc.number_of_nodes()} nodes and {G_largest_cc.number_of_edges()} edges in the largest connected component graph.")
    
    node_percentage = (G_largest_cc.number_of_nodes() / G.number_of_nodes()) * 100
    edge_percentage = (G_largest_cc.number_of_edges() / G.number_of_edges()) * 100
    print(f"The largest connected component contains {node_percentage:.2f}% of the nodes and {edge_percentage:.2f}% of the edges of the original graph.")