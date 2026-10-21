# 3D Computer Vision

Course by Pascal Monasse, Loïc Landrieu and Arslan Artykov. [Course page](https://imagine.enpc.fr/~monasse/Stereo/)

The course is about recovering 3D information and exploring 3D reconstruction from two or more images of a static scene. Intuitively, if the camera only rotates, the images can be registered to build a panorama, but if it also translates, parallax lets us reconstruct the geometry. The course therefore relies on projective geometry, linear algebra and optimization. The labs are in C++ with the Imagine++ library.

## Topics by class

1. **Projective geometry.** Camera model and matrices, homographies and panorama construction.
2. **Epipolar geometry.** Fundamental and essential matrices, how to estimate them, and RANSAC.
3. **Disparity maps.** Stereo matching with correlation.
4. **Graph cuts.** Max-flow / min-cut for disparity.
5. **Multi-view stereo.** Perspective-n-Point (PNP).
6. **Features and deep 3D.** Feature detection and description, deep learning for 3D, NeRFs and Gaussian Splatting.

## Content of this folder

| Lab | Topic | Folder |
| --- | --- | --- |
| Lab 1 | Panorama construction from a homography | [lab1-panorama](lab1-panorama) |
| Lab 2 | Fundamental matrix with RANSAC and epipolar lines | [lab2-fundamental-matrix](lab2-fundamental-matrix) |
| Lab 3 | Disparity map by propagation of seeds | [lab3-seeds-disparity](lab3-seeds-disparity) |
| Lab 4 | Disparity map with graph cuts | [lab4-graph-cut-disparity](lab4-graph-cut-disparity) |

Each folder has the assignment (`exercise.pdf`), my code and a short report (`report.pdf`).

## Lab 1: Panorama

Code: [lab1-panorama](lab1-panorama), report: [report.pdf](lab1-panorama/report.pdf)

The user clicks matching points in two images. From them we estimate the homography between the images by solving a linear system. We then compute the bounding box of the panorama and stitch the images.

## Lab 2: Fundamental matrix

Code: [lab2-fundamental-matrix](lab2-fundamental-matrix), report: [report.pdf](lab2-fundamental-matrix/report.pdf)

We estimate the fundamental matrix F from point matches with RANSAC, using the normalized 8-point algorithm. The result is then refined by least squares on all the inliers. In the end, clicking a point in one image displays its epipolar line in the other one.

## Lab 3: Disparity by propagation of seeds

Code: [lab3-seeds-disparity](lab3-seeds-disparity), report: [report.pdf](lab3-seeds-disparity/report.pdf)

We compute a disparity map for a pair of images. First we keep only the points where the NCC score is very high and use them as seeds in a priority queue. Then the disparity is propagated to the 4-neighbours. 

The figures `0dense`, `1seeds` and `2final` compare the dense result, the seeds and the final map.

## Lab 4: Disparity with graph cuts

Code: [lab4-graph-cut-disparity](lab4-graph-cut-disparity), report: [report.pdf](lab4-graph-cut-disparity/report.pdf)

The disparity map is computed as the minimum of an energy, using a graph cut (max-flow) on a rectified image pair. The energy has a correlation term and a smoothness term weighted by lambda. We compare the result with the propagation method from lab 3 in terms of precision, smoothness and computation time. 

The folder `visualized_results` shows the effect of lambda and of the patch size on several image pairs.

Libraries: C++, Imagine++, and the max-flow library of Boykov and Kolmogorov (included in the lab 4 folder).
