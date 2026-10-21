# Deep Learning for Medical Imaging

Course by Olivier Colliot and Maria Vakalopoulou. [Course page](https://www.master-mva.com/cours/deep-learning-for-medical-imaging/)

The course covers the theory and practice of deep learning for medical images. These images are hard to interpret since they are often 3D, change over time and come from several modalities. The course presents classical image processing and machine learning methods next to modern deep learning, with applications like brain and cardiac imaging. The goal is to be able to build and validate deep learning models for medical images.

## Topics

1. **Introduction to medical imaging.**
2. **Classification.**
3. **Detection.**
4. **Segmentation.**
5. **Registration.**
6. **Denoising and reconstruction.**
7. **Generative models.** Autoencoders and GANs.
8. **Validation, interpretation and reproducibility.**

## Content of this folder

There are no graded labs in this course. The evaluation is an article presentation and a project.

| Project | Topic | Folder |
| --- | --- | --- |
| Project 1 | Zero-shot super-resolution of preclinical MRI (paper review and partial implementation) | [bida-sr](bida-sr) |
| Project 2 | Out-of-distribution histopathology classification (data challenge) | [histopathology-OOD-classification](histopathology-OOD-classification) |

## Project 1: Bidirectional-DDNM-Averaging Super-Resolution

Repository: [vujadinovicn/bida-sr](https://github.com/vujadinovicn/bida-sr)

We first reviewed "Exploring the feasibility of zero-shot super-resolution in preclinical imaging" (Gharib et al., 2024), then implemented parts of it. We implement through-plane super-resolution of rodent brain MRI with unconditional diffusion models trained on two orthogonal planes. At inference, DDNM enforces consistency with the low-resolution measurements and the two reconstructions are averaged. We compare it to cubic interpolation on held-out rat and mouse data.

## Project 2: Histopathology OOD classification

Repository: [vujadinovicn/histopathology-OOD-classification](https://github.com/vujadinovicn/histopathology-OOD-classification)

Our solution to the MVA DLMI 2026 data challenge: tumor vs. non-tumor classification of histopathology patches, where the test patches come from medical centers not seen in training. We start from a frozen DINOv2 baseline and test targeted interventions against the stain and scanner shift: higher resolution, stain-aware augmentation, LoRA fine-tuning, Reinhard normalization, CORAL alignment and the UNI pathology foundation model. Each one is evaluated as an ablation.
