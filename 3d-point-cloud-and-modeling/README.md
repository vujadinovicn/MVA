# 3D Point Cloud and Modeling (NPM3D)

Course by François Goulette, Jean-Emmanuel Deschaud and Tamy Boubekeur. [Course page](https://www.caor.minesparis.psl.eu/presentation/cours-npm3d/)

The course is about the acquisition, processing and visualization of 3D point clouds. It covers 3D perception systems, point cloud operators, registration, local description of curves and surfaces, surface reconstruction, segmentation, primitive-based modeling, rendering and deep learning on point clouds.

## Topics by class

1. **3D perception, sensors and calibration.**
2. **Registration.**
3. **Local description of curves and surfaces.**
4. **Point cloud rendering.** 
5. **Splatting and reconstruction.**
6. **Modeling and segmentation.**
7. **Deep learning for 3D point clouds.**

## Content of this folder

| Lab | Topic | Folder |
| --- | --- | --- |
| Lab 1 | Basic operations on point clouds | [lab1-3d-perception](lab1-3d-perception) |
| Lab 2 | Registration with ICP | [lab2-registration](lab2-registration) |
| Lab 3 | Local descriptors and classification | [lab3-local-descriptors](lab3-local-descriptors) |
| Lab 4 | Surface reconstruction | [lab4-reconstruction](lab4-reconstruction) |
| Lab 5 | Modeling and segmentation with RANSAC | [lab5-modeling-segmentation](lab5-modeling-segmentation) |
| Lab 6 | PointNet | [lab6-deep-learning](lab6-deep-learning) |

Each folder has the code, my report (`report.pdf`) and the assignment (`handout.pdf`).

## Lab 1: Basic operations on point clouds

Code: [lab1-3d-perception](lab1-3d-perception), report: [report.pdf](lab1-3d-perception/report.pdf)

We implement cloud subsampling (decimation and grid subsampling) and neighborhood search, with brute force and with a KD-tree. We analyze how the search time depends on the structure, the leaf size and the radius.

## Lab 2: Registration with ICP

Code: [lab2-registration](lab2-registration), report: [report.pdf](lab2-registration/report.pdf)

We implement the point-to-point ICP algorithm from scratch, with the rigid transform computed from the point correspondences. We use it on 2D and 3D examples and on a real scan, and analyze its convergence through the RMS error and its dependence on the initial position and the subsampling.

## Lab 3: Local descriptors and classification

Code: [lab3-local-descriptors](lab3-local-descriptors), report: [report.pdf](lab3-local-descriptors/report.pdf)

We estimate normals with a local PCA, using a radius and k nearest neighbors, and analyze how the neighborhood scale changes the result. Then we compute geometric descriptors from the PCA (linearity, planarity, sphericity, verticality) and use them to classify an urban LiDAR scan (Lille street) with a random forest.

## Lab 4: Surface reconstruction

Code: [lab4-reconstruction](lab4-reconstruction), report: [report.pdf](lab4-reconstruction/report.pdf)

We implement the Hoppe and IMLS implicit functions to reconstruct a surface from a point cloud with normals (the Stanford bunny). We compare them with Screened Poisson reconstruction in CloudCompare and study the effect of the parameters.

## Lab 5: Modeling and segmentation with RANSAC

Code: [lab5-modeling-segmentation](lab5-modeling-segmentation), report: [report.pdf](lab5-modeling-segmentation/report.pdf)

We implement RANSAC plane detection, then a recursive version that extracts several planes, using the normals to filter the inliers. We use it to segment a scan of an indoor scene into planes.

## Lab 6: PointNet

Code: [lab6-deep-learning](lab6-deep-learning), report: [report.pdf](lab6-deep-learning/report.pdf)

We implement PointNet in PyTorch for point cloud classification on ModelNet. We start with the basic version, then add the T-Net alignment network, and finally try our own 3D data augmentation.
