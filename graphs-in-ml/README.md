# Graphs in Machine Learning

Course by Daniele Calandriello, Michal Valko and Achraf Azize. [Course page](https://www.master-mva.com/cours/graphs-in-machine-learning/)

The course is about machine learning on graphs, either networks (social, biological, technological) or graphs built from unstructured data such as images. It covers spectral clustering, manifold learning, semi-supervised learning and online decision making on graphs, with applications to recommender systems and online advertising. Scalability and the theoretical justification of the methods are emphasized throughout.

## Topics

1. **Spectral graph theory.** Graph Laplacians and spectral clustering.
2. **Building graphs from flat data.**
3. **Semi-supervised and manifold learning.**
4. **Learnability on graphs.** Transductive learning.
5. **Online learning on graphs.**
6. **Scalability.** Approximations, sparsification and error analysis for large graphs.
7. **Decision making on graphs.** Graph bandits.
8. **Applications.** Social networks, recommender systems and vision, like face recognition.

## Content of this folder

| Lab | Topic | Folder |
| --- | --- | --- |
| TD1 | Spectral clustering | [td1-spectral-clustering](td1-spectral-clustering) |
| TD2 | Semi-supervised learning | [td2-semi-supervised-learning](td2-semi-supervised-learning) |
| TD3 | Graph neural networks | [td3-graph-neural-networks](td3-graph-neural-networks) |

## TD1: Spectral clustering


We build spectral clustering step by step and test it on toy datasets and on images.
- **Data and similarity graphs.** We generate datasets like blobs and two moons, then build k-NN and epsilon graphs from them. The questions are about how the parameters change the graph and why a connected graph matters.
- **Laplacian and clustering.** We compute the graph Laplacian, take its eigenvectors, and assign clusters with thresholding or k-means. We also pick the number of clusters automatically and compare spectral clustering to k-means.
- **Parameter sensitivity.** We test how stable the result is when the graph parameters change, and how to evaluate a clustering without the true labels.
- **Image segmentation.** We apply the same method to small images, with a discussion on how to scale it, since a graph on all the pixels gets big fast.

Libraries: numpy, scipy, scikit-learn, networkx, scikit-image.

## TD2: Semi-supervised learning

The goal is to label a graph from only a few known labels, using the harmonic function solution (HFS).
- **HFS.** We implement the hard and the soft version of HFS, and test them on two moons. The questions look at the effect of the number of labels and of noisy labels.
- **Face recognition.** We extend HFS to several classes, tested on a small and an augmented face dataset.
- **Online SSL.** The same idea in an online setting with a webcam-style pipeline, including how to flag a face as unknown.
- **Comparisons.** Ablations on supervised against semi-supervised, inductive against transductive, and batch against online learning.

Libraries: numpy, scipy, scikit-learn, OpenCV.

## TD3: Graph neural networks


An introduction to GNNs with JAX and jraph. We start from how a graph is represented, then build the layers ourselves.
- **GCN.** From a simple message passing layer up to the full GCN with self-edges and symmetric normalization. Node classification on the Karate Club graph.
- **GAT.** A graph attention layer, trained on the same Karate Club task.
- **Graph classification.** Predicting a property of whole molecules on the MUTAG dataset.
- **Link prediction.** Predicting missing edges in the Cora citation network.
- **Open questions.** Ablations on adversarial attacks on graphs and on the Open Graph Benchmark.

Libraries: jax, jraph, networkx.
