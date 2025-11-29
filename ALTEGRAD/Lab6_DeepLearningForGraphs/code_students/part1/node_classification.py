"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
##################
node_color = ['blue' if label == y[0] else 'red' for label in y]
nx.draw_networkx(G, node_color=node_color, with_labels=True)
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim) # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

##################
# your code here #
##################
clf_lr_dw = LogisticRegression()
clf_lr_dw.fit(X_train, y_train)
y_pred_dw = clf_lr_dw.predict(X_test)
accuracy_dw = accuracy_score(y_test, y_pred_dw)

############## Task 8
# Generates spectral embeddings

##################
# your code here #
##################
A = nx.adjacency_matrix(G)
inv_D = diags([1/deg for _, deg in G.degree()], format='csr') # create diagonal degree matrix; use csr for better efficiency
I = eye(G.number_of_nodes(), format='csr') # use csr for better efficiency
L = I - inv_D @ A

# compute the k smallest eigenvectors of L
_, eigenvectors = eigs(L, k=2, which='SR')
spectral_embeddings = np.real(eigenvectors)

X_train_spec = spectral_embeddings[idx_train, :]
X_test_spec = spectral_embeddings[idx_test, :]

clf_lr_spec = LogisticRegression()
clf_lr_spec.fit(X_train_spec, y_train)
y_pred_spec = clf_lr_spec.predict(X_test_spec)
accuracy_spec = accuracy_score(y_test, y_pred_spec)

print(f"Deepwalk embeddings accuracy: {accuracy_dw} vs Spectral embeddings accuracy: {accuracy_spec}")