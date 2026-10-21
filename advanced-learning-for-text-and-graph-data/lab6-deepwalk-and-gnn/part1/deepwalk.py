"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import numpy as np
import networkx as nx
from random import randint, shuffle
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    ##################
    # your code here #
    ##################
    walk = [node]
    for _ in range(walk_length-1):
        neighbors = list(G.neighbors(walk[-1]))
        if not neighbors:
            break
        walk.append(np.random.choice(neighbors))
	
    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    ##################
    # your code here #
    ##################
    for node in list(G.nodes()):
        for _ in range(num_walks):
            node_walk = random_walk(G, node, walk_length)
            walks.append(node_walk)
    
    permuted_walks = walks
    shuffle(permuted_walks)
    return permuted_walks


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
