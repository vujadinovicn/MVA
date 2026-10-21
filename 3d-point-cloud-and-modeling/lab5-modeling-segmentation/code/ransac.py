#
#
#      0===========================================================0
#      |                      TP6 Modeling                         |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from sklearn.neighbors import KDTree



#------------------------------------------------------------------------------------------
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
    cov_matrix = centered_points.T @ centered_points / points.shape[0]

    return np.linalg.eigh(cov_matrix)


def compute_normals(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    nghbrd_search: str = "spherical",
    radius=None,
    k=None
):
    kdtree = KDTree(cloud_points)

    neighborhoods = (
        kdtree.query_radius(query_points, radius)
        if nghbrd_search.lower() == "spherical"
        else kdtree.query(query_points, k=k, return_distance=False)
        if nghbrd_search.lower() == "knn"
        else None
    )

    normals = np.zeros((query_points.shape[0], 3))

    for i in range(len(query_points)):
        normals[i] = PCA(cloud_points[neighborhoods[i]])[1][:, 0]

    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    return normals

def compute_plane(points):
    point = points[0].reshape((3, 1))
    normal = np.cross(points[1] - point.T, points[2] - point.T).reshape((3, 1))

    return point, normal / np.linalg.norm(normal)


def in_plane(points, pt_plane, normal_plane, threshold_in=0.1, normals=None, max_angle=0.1):
    indexes = np.zeros(len(points), dtype=bool)
    dists = np.abs((points - pt_plane.T) @ normal_plane)
    indexes = (dists < threshold_in).squeeze()

    if normals is not None:
        cosines = np.abs((normals @ normal_plane).squeeze())
        angles = np.arccos(np.clip(cosines, -1, 1))
        indexes = indexes & (angles < max_angle)
    return indexes


def RANSAC(points, nb_draws=100, threshold_in=0.1, normals=None, max_angle=0.1):
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    n_points = len(points)

    for _ in range(nb_draws):
        idx = np.random.choice(n_points, size=3, replace=False)
        sample = points[idx]

        pt_plane, normal_plane = compute_plane(sample)
        inliers = in_plane(points, pt_plane, normal_plane, threshold_in=threshold_in, normals=normals, max_angle=max_angle)
        vote = int(np.sum(inliers))

        # Keep best
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane

    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, normals=None, max_angle=0.1):
    
    nb_points = len(points)
    plane_indices = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_indices = np.arange(0,nb_points)
	
    for label in range(nb_planes):
        if len(remaining_indices) < 3:
            break

        sub_normals = None if normals is None else normals[remaining_indices]
        ref, normal, vote = RANSAC(points[remaining_indices], nb_draws, threshold_in, normals=sub_normals, max_angle=max_angle)
        pts_in_plane = in_plane(points[remaining_indices], ref, normal, threshold_in, normals=sub_normals, max_angle=max_angle)

        plane_indices = np.append(plane_indices, remaining_indices[pts_in_plane])
        plane_labels = np.append(plane_labels, np.repeat(label, pts_in_plane.sum()))
        remaining_indices = remaining_indices[~pts_in_plane]

    return plane_indices, remaining_indices, plane_labels




#------------------------------------------------------------------------------------------
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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    # nb_draws = 100
    # threshold_in = 0.10

    # # Find best plane by RANSAC
    # t0 = time.time()
    # best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    # t1 = time.time()
    # print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # # Find points in the plane and others
    # points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    # plane_inds = points_in_plane.nonzero()[0]
    # remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # # Save the best extracted plane and remaining points
    # write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    # write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.10
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    
    print('Done')
    