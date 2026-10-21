#
#
#      0=============================0
#      |    TP3 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):
    barycenter = points.mean(axis=0)
    centered_points = points - barycenter
    cov = centered_points.T.dot(centered_points)/points.shape[0]
    eigen = np.linalg.eigh(cov)

    eigenvalues = eigen[0]
    eigenvectors = eigen[1]

    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius=None, k=None):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    if not radius and not k:
        raise Exception("You must choose between spherical or knn search!")
    tree = KDTree(cloud_points)

    if radius:
        neighbor_indices = tree.query_radius(query_points, radius)
    else:
        neighbor_indices = tree.query(query_points, k=k, return_distance=False)

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for i in range(len(query_points)):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[neighbor_indices[i]])

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    all_eigenvalues, all_eigenvectors = compute_local_PCA(
        query_points, cloud_points, radius=radius
    )
    lbd3, lbd2, lbd1 = (
        all_eigenvalues[:, 0],
        all_eigenvalues[:, 1],
        all_eigenvalues[:, 2],
    )
    lbd1 += 1e-6

    normals = all_eigenvectors[:, :, 0]

    verticality = 2 * np.arcsin(np.abs(normals[:, 2])) / np.pi
    linearity = 1 - lbd2 / lbd1
    planarity = (lbd2 - lbd3) / lbd1
    sphericity = lbd3 / lbd1

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    import numpy as np
    import matplotlib.pyplot as plt
    cloud_path = '../data/Lille_street_small.ply'
    cloud_ply = read_ply(cloud_path)
    cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

    radii = [0.1, 0.5, 1.0, 2.0]
    means_l3, medians_l3, frac_planar = [], [], []
    diffl1l3, diffl2l3 = [], []

    for r in radii:
        ev, evec = compute_local_PCA(cloud, cloud, radius=r)
        l3 = ev[:, 0]                 # smallest eigenvalue = plane fit error
        l2 = ev[:, 1]
        l1 = ev[:, 2] + 1e-6

        planarity = (l2 - l3) / l1

        means_l3.append(np.mean(l3))
        medians_l3.append(np.median(l3))
        # frac_planar.append(np.mean(planarity > 0.6))  # arbitrary threshold, adjust if needed
        frac_planar.append(np.mean(planarity))  # arbitrary threshold, adjust if needed

        diffl1l3.append(np.mean(l1) - np.mean(l3))
        diffl2l3.append(np.mean(l2) - np.mean(l3))
    plt.figure()
    plt.plot(radii, means_l3, marker='o')
    plt.xlabel("Radius (m)")
    plt.ylabel("Mean lambda3")
    plt.grid(True)
    plt.title("Mean of lambda3 vs radius")
    plt.show()

    plt.figure()
    plt.plot(radii, frac_planar, marker='o')
    plt.xlabel("radius (m)")
    plt.ylabel("Mean planarity")
    plt.grid(True)
    plt.title("Mean planarity vs radius")
    plt.show()

    plt.figure()
    plt.plot(radii, diffl1l3, marker='o')
    plt.xlabel("radius (m)")
    plt.ylabel("mean l1 - l3")
    plt.xscale("log")
    plt.grid(True)
    plt.title("Mean difference between l1 and l3 vs radius")

    plt.figure()
    plt.plot(radii, diffl2l3, marker='o')
    plt.xlabel("radius (m)")
    plt.ylabel("mean l2 - l3")
    plt.xscale("log")
    plt.grid(True)
    plt.title("Mean difference between l2 and l3 vs radius")
    plt.show()

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute Spherical PCA on the whole cloud
        spherical_eigenvalues, sperhical_eigenvectors = compute_local_PCA(cloud, cloud, radius=0.50)
        spherical_normals = sperhical_eigenvectors[:, :, 0]

        print(spherical_eigenvalues)
        # Save cloud with normals
        # write_ply('../Lille_street_small_spherical_normals.ply', (cloud, spherical_normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
        # # Compute PCA on the whole cloud
        # knn_eigenvalues, knn_eigenvectors = compute_local_PCA(cloud, cloud, k=30)
        # knn_normals = knn_eigenvectors[:, :, 0]

        # # Save cloud with normals
        # write_ply('../Lille_street_small_knn_normals.ply', (cloud, knn_normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    # if True:
    #     verticality, linearity, planarity, sphericity = compute_features(
    #         cloud, cloud, 0.5
    #     )

    #     print(cloud.shape)
    #     print(verticality.shape, verticality.dtype)
    #     print(linearity.shape, linearity.dtype)
    #     print(planarity.shape, planarity.dtype)
    #     print(sphericity.shape, sphericity.dtype)

    #     write_ply(
    #         "../Lille_street_small_normals_feats.ply",
    #         [cloud, verticality, linearity, planarity, sphericity],
    #         ["x", "y", "z", "verticality", "linearity", "planarity", "sphericity"],
    #     )
