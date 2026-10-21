#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0],1))

    pm_data = np.mean(data, axis=1, keepdims=True)
    pm_ref = np.mean(ref, axis=1, keepdims=True)

    qm_data = data - pm_data
    qm_ref = ref - pm_ref

    H = qm_data.dot(qm_ref.T)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = Vt.T.dot(U.T)

    T = pm_ref - R.dot(pm_data)

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold, subsample_points=None):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    rms = float('inf')
    tree = KDTree(ref.T)

    R_global = np.eye(data.shape[0])
    T_global = np.zeros((data.shape[0], 1))

    for i in range(max_iter):
        distances, indices = tree.query(data_aligned.T, k=1)
        indices = indices.flatten()
        neighbors_list.append(indices.copy())

        rms = float(np.sqrt(np.mean(distances**2)))
        RMS_list.append(rms)
        
        current_data = data_aligned
        current_neighbors = ref[:, indices]

        R, T = best_rigid_transform(current_data, current_neighbors)
        data_aligned = R @ data_aligned + T

        R_global = R @ R_global
        T_global = R @ T_global + T

        R_list.append(R_global.copy())
        T_list.append(T_global.copy())

        if rms < RMS_threshold:
            print("Rms threshold reached at iteration {}".format(i))
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit):
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    rms = float('inf')
    # tree = KDTree(ref.T)

    R_global = np.eye(data.shape[0])
    T_global = np.zeros((data.shape[0], 1))

    from scipy.spatial import cKDTree as KDTree

    ref = ref.astype(np.float32, copy=False)
    data_aligned = data_aligned.astype(np.float32, copy=False)

    tree = KDTree(ref.T)

    store_neighbors = False  # turn on only if you truly need it
    import time 
    ttt = time.time()
    for i in range(max_iter):
        print(i)

        N = data_aligned.shape[1]
        m = min(sampling_limit, N)
        sample_idx = np.random.choice(N, size=m, replace=False)

        # sampled query (parallel)
        dist_s, idx_s = tree.query(data_aligned[:, sample_idx].T, k=1, workers=-1)
        idx_s = idx_s.astype(np.int64, copy=False).ravel()

        current_data = data_aligned[:, sample_idx]
        current_neighbors = ref[:, idx_s]

        R, T = best_rigid_transform(current_data, current_neighbors)
        data_aligned = R @ data_aligned + T

        R_global = R @ R_global
        T_global = R @ T_global + T
        R_list.append(R_global.copy())
        T_list.append(T_global.copy())

        print("ovdesam")
        # FULL query for RMS (parallel)
        dist_all, idx_all = tree.query(data_aligned.T, k=1, workers=-1)

        rms = float(np.sqrt(np.mean(dist_all**2)))
        # rms = np.sqrt(np.mean(np.linalg.norm(data_aligned - ref[idx_all], axis=0)))
        RMS_list.append(rms)

        if rms < RMS_threshold:
            print(f"Rms threshold reached at iteration {i}")
            break

        if i == 0:
            print(time.time()-ttt)


    return data_aligned, R_list, T_list, neighbors_list, RMS_list

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted

    if True:
        notre_dame_1_path = "../data/Notre_Dame_Des_Champs_1.ply"
        notre_dame_2_path = "../data/Notre_Dame_Des_Champs_2.ply"


        ref2D_ply = read_ply(notre_dame_1_path)
        data2D_ply = read_ply(notre_dame_2_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y'], ref2D_ply['z']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y'], data2D_ply['z']))  

        for sampling_limit in [50000]:
            (notre_dame_opt, _, __, ___, RMS_list,) = icp_point_to_point_fast(
                data2D,
                ref2D,
                max_iter=100,
                RMS_threshold=1e-4,
                sampling_limit=sampling_limit,
            )

            plt.plot(RMS_list, label=f"Sampling points limit: {sampling_limit}")

            plt.title("RMS during ICP for Notre Dame Des Champs clouds")
            plt.xlabel("iterations")
            plt.ylabel("RMS")
            plt.legend()

            plt.savefig(f"rms_icp_notre_dame_des_champs_{sampling_limit}.png", dpi=300, bbox_inches="tight")
        # plt.show()  # optional — remove if you don’t want it displayed


    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:}'.format(RMS_before))
        print(' After = {:}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 11, 1e-4)
        
        print(RMS_list)
        # Show ICP
        # show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.title("RMS during ICP for 2D clouds")
        plt.xlabel("iterations")
        plt.ylabel("RMS")
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 26, 1e-4)
        
        # Show ICP
        # show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        # plt.plot(RMS_list)
        # plt.title("RMS during ICP for Bunny clouds")
        # plt.xlabel("iterations")
        # plt.ylabel("RMS")
        # plt.show()

