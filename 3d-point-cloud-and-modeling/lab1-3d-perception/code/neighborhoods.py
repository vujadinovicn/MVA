#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        dists = np.linalg.norm(query - supports, axis=1)
        indices = np.where(dists < radius)[0]
        neighborhoods.append(supports[indices])

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = []
    for query in queries:
        dists = np.linalg.norm(query - supports, axis=1)
        indices = np.argpartition(dists, k)[:k]
        neighborhoods.append(supports[indices])

    return neighborhoods





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:
    #     # YOUR CODE
    #     # setup of data
        num_queries = 1000
        radius = 0.2

        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # optimal leaf size 4a
        leaf_sizes = range(1, 100) # check for all leaves between 1-100
        times = []
        best_leaf = 1
        best_time = float('inf')

        for leaf_size in leaf_sizes:
            tree = KDTree(points, leaf_size=leaf_size)

            t0 = time.time()
            indices = tree.query_radius(queries, r=radius)
            neighborhoods = points[indices[0]]
            t = time.time() - t0

            times.append(t)
            if t < best_time:
                best_leaf = leaf_size
                best_time = t

        print(f"Best leaf_size: {best_leaf} and best query time: {best_time} seconds")

        plt.figure()
        plt.plot(list(leaf_sizes), times)
        plt.xlabel("Leaf size")
        plt.ylabel("Query time (s)")
        plt.title(f"KDTree spherical query time vs leaf size (1000 queries, radius={radius})")
        plt.show()

        # timings as function of radius 4b
        radiuses = np.linspace(0.01, 1, 20)
        times = []
        for radius in radiuses:
            tree = KDTree(points, leaf_size=best_leaf)

            t0 = time.time()
            indices = tree.query_radius(queries, r=radius)
            neighborhoods = points[indices[0]]
            t = time.time() - t0

            times.append(t)

        plt.figure()
        plt.plot(list(radiuses), times)
        plt.xlabel("Radius (m)")
        plt.ylabel("Query time (s)")
        plt.title(f"KDTree spherical query time vs radius (1000 queries)")
        plt.show()

        tree = KDTree(points, leaf_size=best_leaf)

        t0 = time.time()
        indices = tree.query_radius(queries, r=0.2)
        neighborhoods = points[indices[0]]
        t = time.time() - t0

        print(f"Time needed for whole point cloud query search and radius of {radius} m is: {points.shape[0]*t/num_queries} seconds")





        
        
        
        