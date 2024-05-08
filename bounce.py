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
    def angle(dir):
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
        #dir2 = dir[1:] # all excluding the last
        #dir1 = dir[:-1] # all excluding the first
        dir2 = dir[1:]
        dir1 = dir[:-1]
        radians = np.arccos(
            (dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1)))
                         )

        degrees = np.degrees(radians)
        return degrees

    """Eliminates groups of redundant points from indices, as not to draw too many crosses."""
    """This is done by taking, among all the identified points, the middle one. Better approaces will be found."""
    def eliminate_redundant_points(x, y, sx, sy, ix):
        def cluster(data, maxgap):
            '''Arrange data into groups where successive elements
            differ by no more than *maxgap*

                >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
                [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

                >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
                [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

            '''
            data.sort()
            groups = [[data[0]]]
            for x in data[1:]:
                if abs(x - groups[-1][-1]) <= maxgap:
                    groups[-1].append(x)
                else:
                    groups.append([x])
            return groups
        
        def get_min(groups):
            points = []
            min = 100000
            min_idx = -1

            for group in groups:
                
                for idx in group:
                    if y[idx] < min:
                        min = y[idx]
                        min_idx = idx
                
                min = 100000
                points.append(min_idx)
                min_idx = -1

            return points
        
        def get_midpoint(groups):
            mps = []
            for group in groups:
                i = 1-int(np.ceil(len(group)/2)) # Group centroid
                mps.append(group[i])
            
            return mps

        groups = cluster(ix, 1)
        
        new_indices = get_midpoint(groups)
        # new_indices = get_min(groups)

        # Remove points if the y axis before and after are lower (using < for convention)
        """
        buf = []
        for j in range(len(new_indices)-1):
            interest = new_indices[j]
            prev_neigh = interest-1
            next_neigh = interest+1
            if (y[interest] <= y[prev_neigh]) and (y[interest] <= y[next_neigh]):
                # print( "Found parabola peak: " + str(x[i]) + ", " + str(y[i]) )
                # print("index: " + str(i))
                buf.append(interest)
        s = set(buf)        
        result = [x for x in new_indices if x not in s]
        """
        
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
    theta = angle(directions)
    print(theta)

    # Select the index of the points (in the simplified trajectory) with the greatest theta
    # Large theta is associated with greatest change in direction.
    idx_simple_trajectory = np.where(theta>min_angle)[0]+1
    # theta_new = theta[idx_simple_trajectory]
    # idx_simple_trajectory = np.where(theta_new<max_angle)[0]+1

    # Return real indices of bouncing points
    ix = find_indices(points, list(zip(sx, sy)), idx_simple_trajectory)

    # Filter redundant points via clustering
    ix = eliminate_redundant_points(x, y, sx, sy, ix)
    # print(ix)
    
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
