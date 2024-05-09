import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
import scipy.io
import pathlib
from pathlib import Path
import rdp

"""Helper function to draw crosses on each frame if a bounce is detected."""
def draw_cross(frame, x, y, size=10, color=(255, 0, 0), thickness=2):
    if not math.isnan(x):
        x, y = int(x), int(y)  # Convert coordinates to integers
        cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
        cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness)

"""Parse input. A sample command is in the README."""
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str, help='path to csv')
    parser.add_argument('--path_to_video', type=str, help='path to inferred video')
    parser.add_argument('--path_to_output_video', type=str, help='path to save the video with marked bounces')
    return parser.parse_args()

"""Drawing on video."""
def draw_video(ix, args):
    if os.path.exists(args.path_to_video):
        video_capture = cv2.VideoCapture(args.path_to_video)
    else:
        print("The specified file does not exist.")

    # Video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.path_to_output_video, fourcc, fps, (width, height))

    # Variables for storing frames where a bounce is detected
    i = 0
    visited = []
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # If at that frame a bounce is detected, mark the bounce as visited
        if i in ix:
            visited.append(i)

        # Draw crosses at the specified (x, y) positions every frame onwards from bounce detection
        for item in visited:
            draw_cross(frame, x[item], y[item])

        out.write(frame)
        i = i+1

    video_capture.release()
    out.release()

"""With simple angle checking"""
def bounce_angle_wrapping(x, y, args):
    
    """Plotting bounces"""
    def plot_bounces(x, y, ix, dt):
        plt.plot(x[ix + 1], y[ix + 1], 'or')  # Plotting points where angle difference > 150
        plt.plot(x, y)  
        for i in range(len(ix)):
            plt.text(x[ix[i] + 1], y[ix[i] + 1], str(dt[ix[i]]))
        plt.show()

    t = np.degrees(np.arctan2(np.diff(y), np.diff(x))) # Calculate angle of each line
    dt = np.degrees(np.diff(t)) # Calculate angle between lines
    dt = (dt + 180) % 360 - 180  # Wrap angles to [-180, 180)

    # Find indices where angle difference > 150
    ix = np.where(np.abs(dt) > 150)[0]

    plot_bounces(x, y, ix, dt)

    draw_video(ix, args)

"""With Ramer-Douglas-Pecker algorithm"""
def rdp_algo(x, y, args):

    """Finds indices of bounces in the points list instead of the simplified trajectory."""
    def find_indices(points, sp, idx):
        ix = []

        for index in idx:
            ix.append(points.index(sp[index]))

        return ix
    
    """Finds angles between lines in the simplified trajectory."""
    def angle(dir, points):
        """
        Returns the angles between vectors.

        Parameters:
        dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space. (Tot directions in 2D space).

        The return value is a 1D-array of values of shape (N-1,), with each value
        between 0 and pi.

        0 implies the vectors point in the same direction
        pi/2 implies the vectors are orthogonal
        pi implies the vectors point in opposite directions
        """
        dir2 = dir[1:]
        dir1 = dir[:-1]
        radians = np.arccos(
            (dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1)))
                         )

        degrees = np.degrees(radians)
        
        # Rebuild vectors for quandrant check
        vectors = []
        print(points.shape)
        for i in range(len(points) - 2):
            p1 = (points[i])
            p2 = (points[i + 1])
            p3 = (points[i + 2])

            vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            vectors.append((vector1, vector2))               

        # Quadrant check
        is_bounce = []
        for i in range(0, len(vectors)):
            # Vectors
            first = np.array(vectors[i][0])
            second = np.array(vectors[i][1])
            
            print("Vector pairs: ", first, second)
            angle_rad = np.arctan2(np.linalg.norm(np.cross(first, second)), np.dot(first, second))
            angle_deg = np.degrees(angle_rad)
            #if np.linalg.norm(second) > 20:
            is_bounce.append(not (angle_deg <= 90))
            #else:
            #    is_bounce.append(False)
            
            
        return degrees, is_bounce

    """Eliminates groups of redundant points from indices, as not to draw too many crosses."""
    """This is done by taking the centroid of the cluster."""
    def eliminate_redundant_points(x, y, sx, sy, ix):
        def cluster_by_time(ix, maxgap):
            '''Arrange data into groups where successive elements
            differ by no more than *maxgap*

                >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
                [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

                >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
                [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

            '''
            ix.sort()
            groups = [[ix[0]]]
            for x in ix[1:]:
                if abs(x - groups[-1][-1]) <= maxgap:
                    groups[-1].append(x)
                else:
                    groups.append([x])
                    
            
            return groups
        
        def cluster_by_space(groups, x, y):
            print(groups)
            new_groups = []
            for group in groups:
                points = []
                new_group = [group[0]] # a new group that has as a first item the first element of the current group, which is an index

                for i in group: # all points in the current group
                    points.append(np.array(x[i], y[i]))

                for i in range(len(points)-1):
                    gap = np.linalg.norm(points[i+1]-points[i]) # gap between 2 points in a group
                    if gap > 10: # if over a certain threshold
                        new_groups.append(new_group) # I need to create a new group 
                        new_group = [group[i+1]] 
                    else:
                        new_group.append(group[i+1])
                
                new_groups.append(new_group)
            print(new_groups)
            return new_groups
        
        def get_midpoint(groups):
            mps = []
            for group in groups:
                i = 1-int(np.ceil(len(group)/2)) # Group centroid
                mps.append(group[i])
            
            return mps

        groups = cluster_by_time(ix, 1)
        #groups = cluster_by_space(groups, x, y)
        new_indices = get_midpoint(groups)

        #new_groups = cluster(new_indices, 8)

        #new_indices = get_first(new_groups)
        
        return new_indices

    """Plots bouncing points."""
    def plot_bounces(x, y, sx, sy, ix):
        fig = plt.figure()
        ax =fig.add_subplot(111)

        # ax.plot(x, y, 'b-', label='original path')
        ax.plot(sx, sy, 'g--', label='simplified path')
        ax.plot(x[ix], y[ix], 'ro', markersize = 10, label='turning points')

        # Displays indices from the normal trajectory
        i = 0
        for x, y in zip(x[ix], y[ix]):
            ax.text(x, y, str(ix[i]), color="black", fontsize=8)
            i = i+1
        
        ax.invert_yaxis()
        plt.legend(loc='best')
        plt.show()
        
    tolerance = 5 # a normal value is 70
    min_angle = 25 # min angle = np.pi*0.15 works fine

    points = list(zip(x.to_list(), y.to_list()))

    # Use the Ramer-Douglas-Peucker algorithm to simplify the path
    # http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
    # Python implementation: https://github.com/sebleier/RDP/
    simplified = np.array(rdp.rdp(points, tolerance))

    # print(len(simplified))
    sx, sy = simplified.T
    # print(sx, sy)

    # compute the direction vectors on the simplified curve
    directions = np.diff(simplified, axis=0)
    print(directions)
    # Determine the quadrants each vector lies in
    # quadrants_directions = np.sign(directions)

    # Check if the angle is predominantly in the first or fourth quadrant
    theta, is_bounce = angle(directions, simplified)
    # Select the index of the points (in the simplified trajectory) with the greatest theta
    # Large theta is associated with greatest change in direction.
    idx_simple_trajectory = np.where(theta>min_angle)[0]+1

    idx_filtered = []
    for index in idx_simple_trajectory:
        if index+1 < len(is_bounce):
            if is_bounce[index] == True:
                idx_filtered.append(index)

    # Return real indices of bouncing points
    ix = find_indices(points, list(zip(sx, sy)), idx_filtered)

    # Filter redundant points via clustering
    ix = eliminate_redundant_points(x, y, sx, sy, ix)
    
    plot_bounces(x, y, sx, sy, ix)

    draw_video(ix, args)

"""Velocity based approach. DO NOT USE."""
def velocity_approach(x, y, args): 

    # Function to calculate velocity
    def calculate_velocity(x, y, dt):
        dx = np.diff(x)
        dy = np.diff(y)
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / dt
        return velocity

    # time = df.timestamp  # Time points

    # Calculate time differences
    # dt = np.diff(time)

    # Calculate velocity
    velocity = calculate_velocity(x, y, 3) # 3rd param is dt

    # Detect abrupt changes (threshold for velocity change)
    threshold = 10  # Adjust as needed
    ix = np.where(np.abs(np.diff(velocity)) > threshold)[0]
    print(ix)

    # Plot trajectory
    plt.plot(x, y, '-o')

    # Mark abrupt changes
    if ix.size > 0:
        plt.plot(x[ix+1], y[ix+1], 'rx', markersize=10, label='Abrupt change')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory with Abrupt Changes')
    plt.grid(True)
    plt.show()

    draw_video(ix, args)

# Height of net in the middle: 3 ft
# Court height and width: 78 ft, 36 ft
# A: 0, 0
# B: 0, 36
# C: 78, 0
# D: 78, 36
# E: net middle point

# Find line at the infnity
if __name__ == '__main__':

    # Parse input
    args = parse()

    # Obtain coordinates from csv
    columns = ['x', 'y']
    df = pd.read_csv(args.path_to_csv, usecols=columns)
    x = df.x
    y = df.y

    # Detection methods

    # bounce_angle_wrapping(x, y, args)    
    rdp_algo(x, y, args)
    # velocity_approach(x, y, args)
