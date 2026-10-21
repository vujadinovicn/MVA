# Geometry Processing and Geometric Deep Learning

Course by Étienne Corman, Julie Digne and Maks Ovsjanikov. [Course page](https://jdigne.github.io/mva_geom/)

The course is about analysing 3D geometry, mostly surfaces and meshes. The first part is classical geometry processing, with discrete differential geometry, spectral methods and surface mappings. The second part is deep learning on 3D data, from point clouds and meshes to neural fields and generative models for shapes.

## Topics by class

1. **Discrete differential geometry.** Differential operators on surfaces, FEM discretization, geodesics, integration and convolution on surfaces.
2. **Spectral methods and mappings.** Curvature, shape deformation, geometric energy optimization, surface parameterization, maps between surfaces and surface topology.
3. **3D deep learning.** Multi-view and volumetric approaches, then learning directly on point clouds.
4. **Intrinsic learning on surfaces.** Geodesic CNNs and learning via diffusion (DiffusionNet).
5. **Neural fields.** Representing, generating and analysing surfaces with DeepSDF and occupancy networks.
6. **Generative models for shapes.** Latent shape spaces and novel view synthesis.

## Content of this folder

| Lab | Topic | Folder |
| --- | --- | --- |
| TD2 | Surface parameterization and alignment (Tutte, LSCM, ICP, ARAP) | [td2-parameterization-and-alignment](td2-parameterization-and-alignment) |
| TD4 | DiffusionNet for RNA surface segmentation | [td4-diffusionnet-rna-segmentation](td4-diffusionnet-rna-segmentation) |
| TD6 | Implicit shape interpolation with SIREN | [td6-implicit-shape-interpolation](td6-implicit-shape-interpolation) |
| Project | Paper review of Zero-1-to-3 (Liu et al., ICCV 2023) | [zero-1-to-3-review](zero-1-to-3-review) |

## TD2: Surface parameterization and alignment

Notebook: [td2_parameterization_and_alignment.ipynb](td2-parameterization-and-alignment/td2_parameterization_and_alignment.ipynb)

In the first part we compute a parameterization of a triangle mesh, meaning a 2D coordinate for each vertex. We use it to map a texture on the 3D surface.
- **Tutte embedding.** The border is fixed on a circle and each inner vertex sits at the barycenter of its neighbours. This comes down to solving a sparse linear system.
- **Least Square Conformal Map (LSCM).** An angle preserving map with a free boundary. We build the system, solve it and compare it with Tutte on the same border.

In the second part we align surfaces.
- **ICP.** Iterative Closest Point cloud registration.
- **ARAP (as-rigid-as-possible) deformation.** We alternate between estimating a rotation for each vertex from a covariance matrix and solving for the new vertex positions. At the end we use it to write a simple non-rigid registration algorithm.

Libraries: numpy, scipy (sparse solvers), meshplot.

## TD4: DiffusionNet for RNA segmentation

Notebook: [td4_diffusionnet_rna_segmentation.ipynb](td4-diffusionnet-rna-segmentation/td4_diffusionnet_rna_segmentation.ipynb)

Here we implement [DiffusionNet](https://arxiv.org/pdf/2012.00888.pdf) in PyTorch and use it to segment RNA molecules seen as surfaces.

The network is built block by block:
- projection to and from the spectral basis,
- the spectral diffusion module with a learned diffusion time per feature,
- the gradient features module,
- a per-vertex MLP,
- the full DiffusionNet block and then the whole network.

We then train it for 200 epochs and look at the segmentation results. The last part is an ablation study where we change one thing at a time to see what each component brings:
- no diffusion,
- no gradient features,
- HKS input features,
- WKS input features.

The trained weights are in [models](td4-diffusionnet-rna-segmentation/models) and the segmentation results are in [visualization](td4-diffusionnet-rna-segmentation/visualization).

Libraries: PyTorch.

## TD6: Implicit shape interpolation

Notebook: [td6_implicit_shape_interpolation.ipynb](td6-implicit-shape-interpolation/td6_implicit_shape_interpolation.ipynb)

In this lab we learn the path between two shapes as a single neural signed distance function $F(x, t)$. At $t = 0$ it is the SDF of the first shape and at $t = 1$ it is the SDF of the second one. The network used is a [SIREN](https://arxiv.org/pdf/2006.09661).

We write the losses ourselves:
- a data loss, where the network is zero on the shape and its gradient matches the normals,
- an SDF loss against the ground truth distance in the ambient space, like in DeepSDF,
- an eikonal loss, so the gradient norm of the SDF stays at 1,
- a level set equation loss, which makes the shape move along a vector field over time.

We test it on simple 2D shapes by interpolating between a square and a circle. The bonus part tries the same approach on real shapes.

Libraries: PyTorch.

## Project: Zero-1-to-3 review

Repository: [vujadinovicn/zero-1-to-3-review](https://github.com/vujadinovicn/zero-1-to-3-review), slides: [presentation.pdf](zero-1-to-3-review/presentation.pdf)

The final project is the review and presentation of a research paper. We reviewed "Zero-1-to-3: Zero-shot One Image to 3D Object" (Liu et al., ICCV 2023), a camera-conditioned diffusion model that generates new views of an object from a single image. We also wrote a script that runs the released pretrained model to generate six views of an input object (front, right, back, left, top and bottom).
