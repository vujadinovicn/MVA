# Geometric Data Analysis

Course by Jean Feydy. [Course page](https://www.master-mva.com/cours/geometric-data-analysis/)

The course is about a geometric view of machine learning models. It shows how geometry encodes domain knowledge through distances and structured representations, and how this guides the choice of data representation (vectors, graphs or histograms). It covers flat vector spaces, graphs and embeddings, geometric deep learning, Riemannian metrics and geodesics, probability distributions with information geometry and optimal transport, and GPU programming. 

## Topics by class

1. **Introduction.** Curse of dimensionality, geometry as a modeling tool.
2. **Flat vector spaces.** Decision trees, nearest neighbors, linear models, neural networks and kernel methods.
3. **Graphs and embeddings.** Manifold hypothesis, dimension and curvature.
4. **Geometric deep learning.** Graph convolutions and case studies on protein docking and lung registration.
5. **Riemannian metrics and geodesics.** Poincaré disk, shortest paths.
6. **Probability distributions.** Information geometry, kernel norms and optimal transport.
7. **GPU programming.**

## Content of this folder

| Project | Topic | Folder |
| --- | --- | --- |
| Project | A reproduction and extension of "Generative Modeling by Estimating Gradients of the Data Distribution" (Song & Ermon, 2019) | [ScoreBasedGenerativeModels](ScoreBasedGenerativeModels) |

## Project: Score-based generative modeling

Repository: [vujadinovicn/ScoreBasedGenerativeModels](https://github.com/vujadinovicn/ScoreBasedGenerativeModels)

We reproduce and extend "Generative Modeling by Estimating Gradients of the Data Distribution" (Song & Ermon, 2019). We implement a noise-conditional score network with annealed Langevin dynamics in PyTorch from scratch, trained on a subset of Fashion-MNIST. We use it to study the choice of the variance schedule and to generate images in a PCA latent space.
