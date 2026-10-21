# Introduction to Probabilistic Graphical Models and Deep Generative Models

Course by Pierre Latouche, Pierre-Alexandre Mattei and Mathieu Even. [Course page](https://lmbp.uca.fr/~latouche/mva/IntroductiontoProbabilisticGraphicalModelsMVA.html)

The course is about probabilistic modeling with graphical models and the algorithms to learn them and to do inference. It covers clustering and latent variable models, approximate inference (variational methods, MCMC), deep latent variable models, variational autoencoders, energy-based models, score-based diffusion models, GANs and causal inference. 

## Topics by class

1. **Directed graphical models.** Maximum likelihood, linear and logistic regression.
2. **Clustering and latent variables.** K-means, EM, Gaussian mixtures, probabilistic PCA.
3. **Bayesian methods.** Bayesian linear regression, Gaussian processes, model selection.
4. **Approximate inference I.** Variational techniques, stochastic block models, expectation propagation.
5. **Directed graphical models.** Theory and examples.
6. **Approximate inference II.** Amortised variational inference and variational autoencoders.
7. **Undirected models and sampling.** Energy-based models, score-based diffusion models, MCMC.
8. **Causal inference.** Potential outcomes, treatment effects, policy learning.
9. **Beyond likelihood-based models.** GANs and likelihood-free inference.

## Content of this folder

| Project | Topic | Folder |
| --- | --- | --- |
| Project | A student reproduction and extension of "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) | [DDPM](DDPM) |

## Project: Denoising Diffusion Probabilistic Models

We reproduce and extend "Denoising Diffusion Probabilistic Models" (Ho et al., 2020). The DDPM is implemented from scratch in PyTorch and trained on a subset of MNIST. We use it to compare four theoretically equivalent training objectives, to study a learned reverse variance, and to apply classifier-free guidance.
