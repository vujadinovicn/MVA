#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels

def grid_subsampling(points, colors, labels, voxel_size):

    # YOUR CODE
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {} # dict for saving points, colors and labels data in each voxel
    
    for i in range(len(points)):
        # add voxel coordinate to dict
        voxel_coord = tuple(voxel_coords[i])
        if voxel_coord not in voxel_dict: 
            voxel_dict[voxel_coord] = {'points': [], 'colors': [], 'labels': []}
        
        # add data to dict
        voxel_dict[voxel_coord]['points'].append(points[i])
        voxel_dict[voxel_coord]['colors'].append(colors[i])
        voxel_dict[voxel_coord]['labels'].append(labels[i])
    
    subsampled_points, subsampled_colors, subsampled_labels = [], [], []
    
    for voxel_coord, voxel_data in voxel_dict.items():
        # proccess points
        voxel_points = np.array(voxel_data['points'])
        barycenter = np.mean(voxel_points, axis=0)
        subsampled_points.append(barycenter)
        
        # process colors
        voxel_colors = np.array(voxel_data['colors'])
        avg_color = np.mean(voxel_colors, axis=0)
        subsampled_colors.append(avg_color)
        
        # process labels (using majority voting)
        voxel_labels = np.array(voxel_data['labels'])
        unique_labels, counts = np.unique(voxel_labels, return_counts=True)
        most_frequent_label = unique_labels[np.argmax(counts)]
        subsampled_labels.append(most_frequent_label)
    
    return np.array(subsampled_points), np.array(subsampled_colors), np.array(subsampled_labels)





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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.32f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    
    print('Done')

    # subsampling
    if True:
        t0 = time.time()
        subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling(points, colors, labels, voxel_size=0.2)
        write_ply('../grid_decimated.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        t1 = time.time()
        print('subsampling done in {:.7f} seconds'.format(t1 - t0))

        #subsampling done in 12.2384968 seconds