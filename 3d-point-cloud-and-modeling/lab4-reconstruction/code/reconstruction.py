#
#
#      0===========================================================0
#      |              TP5 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    # YOUR CODE    
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    min_grid = np.asarray(min_grid, dtype=np.float64)

    my_tree = KDTree(points)

    d = points.shape[1]
    coords = np.arange(grid_resolution)

    voxels = np.stack(np.meshgrid(*([coords] * d), indexing="ij"), axis=-1)
    voxels = (voxels * size_voxel + min_grid).reshape(-1, d)

    indices = my_tree.query(voxels, k=1, return_distance=False).squeeze()

    volume = np.sum(normals[indices] * (voxels - points[indices]), axis=-1)

    print("grid_resolution:", grid_resolution)
    print("voxels.shape:", voxels.shape)     # should be (grid_resolution**3, 3)
    print("volume.shape:", volume.shape)     # should be (grid_resolution**3,)
    print("scalar_field.shape:", scalar_field.shape)


    scalar_field[:, :, :] = volume.reshape(grid_resolution, grid_resolution, grid_resolution).astype(np.float32)

    # points = np.asarray(points, dtype=np.float64)
    # normals = np.asarray(normals, dtype=np.float64)
    # min_grid = np.asarray(min_grid, dtype=np.float64)

    # # Build KDTree
    # tree = KDTree(points)

    # # Grid node coordinates
    # gx = min_grid[0] + size_voxel * np.arange(grid_resolution, dtype=np.float64)
    # gy = min_grid[1] + size_voxel * np.arange(grid_resolution, dtype=np.float64)
    # gz = min_grid[2] + size_voxel * np.arange(grid_resolution, dtype=np.float64)

    # X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    # grid_pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))  # (N,3)

    # # Nearest neighbor index for each grid node
    # idx = tree.query(grid_pts, k=1, return_distance=False).reshape(-1)  # idx shape: (N,)

    # p_closest = points[idx]
    # n_closest = normals[idx]

    # # Hoppe implicit function
    # f = np.einsum("ij,ij->i", n_closest, (grid_pts - p_closest))

    # scalar_field[:, :, :] = f.reshape((grid_resolution, grid_resolution, grid_resolution)).astype(np.float32)
    # return
    

# IMLS surface reconstruction
# def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
#     # YOUR CODE
#     points = np.asarray(points, dtype=np.float64)
#     normals = np.asarray(normals, dtype=np.float64)
#     min_grid = np.asarray(min_grid, dtype=np.float64)

#     # Build KDTree (default leaf_size)
#     tree = KDTree(points)

#     d = points.shape[1]  # should be 3
#     coords = np.arange(grid_resolution)

#     voxels = np.stack(np.meshgrid(*([coords] * d), indexing="ij"), axis=-1)
#     voxels = (voxels * size_voxel + min_grid).reshape(-1, d)  # (N,3)

#     # kNN query
#     distances, indices = tree.query(voxels, k=knn)  # (N,k)

#     # weights theta = exp(-||x-p||^2 / h^2)
#     theta = np.exp(-((distances / 0.01) ** 2))  # (N,k)

#     # n_i · (x - p_i)
#     hoppe = np.sum(
#         normals[indices] * (voxels[:, None, :] - points[indices]),
#         axis=-1,
#     )  # (N,k)

#     # IMLS implicit function
#     volume = np.sum(hoppe * theta, axis=-1) / (np.sum(theta, axis=-1) + 1e-16)

#     scalar_field[:, :, :] = volume.reshape(
#         grid_resolution, grid_resolution, grid_resolution
#     ).astype(np.float32)


def compute_imls(
    points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn, h=0.01
):
    my_tree = KDTree(points, 10)
    d = points.shape[1]  # should be equal to 3

    coords = np.arange(grid_resolution)
    voxels = np.stack(np.meshgrid(*d * [coords], indexing="ij"), axis=-1)
    voxels = (voxels * size_voxel + min_grid).astype(np.float32).reshape(-1, d)

    distances, indices = my_tree.query(voxels, k=knn)

    # computing theta once for all because it is used twice
    theta = np.exp(-((distances / h) ** 2))
    hoppe = np.sum(
        normals[indices] * (voxels[:, np.newaxis] - points[indices]), axis=-1
    )
    volume = np.sum(hoppe * theta, axis=-1) / (np.sum(theta, axis=-1) + 1e-16)

    scalar_field[:, :, :] = volume.reshape(*d * [grid_resolution])

if __name__ == '__main__':
    times = []
    if False:
        t0 = time.time()
        
        # Path of the file
        file_path = '../data/bunny_normals.ply'

        # Load point cloud
        data = read_ply(file_path)

        # Concatenate data
        points = np.vstack((data['x'], data['y'], data['z'])).T
        normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

        # Compute the min and max of the data points
        min_grid = np.amin(points, axis=0)
        max_grid = np.amax(points, axis=0)
                    
        # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
        min_grid = min_grid - 0.10*(max_grid-min_grid)
        max_grid = max_grid + 0.10*(max_grid-min_grid)

        # grid_resolution is the number of voxels in the grid in x, y, z axis
        grid_resolution = 128 #128
        size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
        print("size_voxel: ", size_voxel)
        
        # Create a volume grid to compute the scalar field for surface reconstruction
        scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

        # Compute the scalar field in the grid
        compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
        #compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

        # Compute the mesh from the scalar field based on marching cubes algorithm
        verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
        verts += min_grid
        
        # Export the mesh in ply using trimesh lib
        mesh = trimesh.Trimesh(vertices = verts, faces = faces)
        mesh.export(file_obj='../bunny_mesh_hoppe_16.ply', file_type='ply')
        
        print("Total time for surface reconstruction : ", time.time()-t0)

        times.append(time.time()-t0)

    for _ in range(1):
        t0 = time.time()
        
        # Path of the file
        file_path = '../data/bunny_normals.ply'

        # Load point cloud
        data = read_ply(file_path)

        # Concatenate data
        points = np.vstack((data['x'], data['y'], data['z'])).T
        normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

        # Compute the min and max of the data points
        min_grid = np.amin(points, axis=0)
        max_grid = np.amax(points, axis=0)
                    
        # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
        min_grid = min_grid - 0.10*(max_grid-min_grid)
        max_grid = max_grid + 0.10*(max_grid-min_grid)

        # grid_resolution is the number of voxels in the grid in x, y, z axis
        grid_resolution = 128 #128
        
        size_voxel = np.array(
            [
                (max_grid[0] - min_grid[0]) / (grid_resolution - 1),
                (max_grid[1] - min_grid[1]) / (grid_resolution - 1),
                (max_grid[2] - min_grid[2]) / (grid_resolution - 1),
            ]
        )
        print("size_voxel: ", size_voxel)
        
        # Create a volume grid to compute the scalar field for surface reconstruction
        scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

        # Compute the scalar field in the grid
        # compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
        compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

        # Compute the mesh from the scalar field based on marching cubes algorithm
        verts, faces, normals_tri, values_tri = measure.marching_cubes(
            scalar_field, level=0.0, spacing=(size_voxel[0], size_voxel[1], size_voxel[2])
        )
        verts += min_grid
        
        # Export the mesh in ply using trimesh lib
        mesh = trimesh.Trimesh(vertices = verts, faces = faces)
        mesh.export(file_obj='../bunny_mesh_imls_16_diff.ply', file_type='ply')
        
        print("Total time for surface reconstruction : ", time.time()-t0)

        times.append(time.time()-t0)

    print("Average time: ", np.mean(times))
        


