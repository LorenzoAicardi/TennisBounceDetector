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
from scipy.interpolate import splprep, splev

def find_intercept(arr1, arr2, x_arr, distance_threshold=10):
    intercept = []
    i, j = 0, 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            intercept.append(arr1[i])
            i += 1
            j += 1
        elif abs(arr1[i] - arr2[j]) == 1:
            intercept.append(arr2[j])
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1

    final_intercept = []
    for index in intercept:
        final_intercept.append(index)
        print("index:")
        print(index)
        idx_in_arr2 = arr2.index(index)
        if idx_in_arr2 > 0 and idx_in_arr2 < len(arr2) - 1:
            
            idx_before = arr2[idx_in_arr2 - 1]
            idx_after = arr2[idx_in_arr2 + 1]
            print("index before:")
            print(idx_before)
            
            print("index after:")
            print(idx_after)
        
            x_value_before = x_arr[idx_before]
            x_value_after = x_arr[idx_after]
            x_value_intercept = x_arr[index]
            if abs(x_value_intercept - x_value_before) > distance_threshold:
                final_intercept.append(idx_before)
            elif abs(x_value_after - x_value_intercept) > distance_threshold:
                
                final_intercept.append(idx_after)
    final_list = list(set(final_intercept))
    original_list= list(set(intercept))
    return final_list, original_list

"""Plots bouncing points."""
def plot_bounces(x, y, sx, sy, ix):
    fig = plt.figure()
    ax =fig.add_subplot(111)

    # ax.plot(x, y, 'b-', label='original path')
    ax.plot(sx, sy, 'g--', label='simplified path')
    ax.plot(x[ix], y[ix], 'ro', markersize = 10, label='bounces')

    # Displays indices from the normal trajectory
    i = 0
    for x, y in zip(x[ix], y[ix]):
        ax.text(x, y, str(ix[i]), color="black", fontsize=8)
        i = i+1
        
    ax.invert_yaxis()
    plt.legend(loc='best')
    plt.show()

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

"""Plotting bounces"""
def plot_bounces_4(x, y, ix, dt):
    plt.plot(x[ix + 1], y[ix + 1], 'or')  # Plotting points where angle difference > 150
    plt.plot(x, y)  
    for i in range(len(ix)):
        plt.text(x[ix[i] + 1], y[ix[i] + 1], str(dt[ix[i]]))
    plt.show()

"""Finds indices of bounces in the points list instead of the simplified trajectory."""
def find_indices(points, sp, idx):
    ix = []
    for index in idx:
        ix.append(points.index(sp[index]))
    return ix

"""With simple angle checking"""
def bounce_angle_wrapping(x, y, args):
    t = np.degrees(np.arctan2(np.diff(y), np.diff(x))) # Calculate angle of each line
    dt = np.degrees(np.diff(t)) # Calculate angle between lines
    dt = (dt + 180) % 360 - 180  # Wrap angles to [-180, 180)

    # Find indices where angle difference > 150
    ix = np.where(np.abs(dt) > 150)[0]

    plot_bounces_4(x, y, ix, dt)

    draw_video(ix, args)

def angle(dir, points):

    """Finds angles between lines in the simplified trajectory."""
    # Vectors for quandrant check
    vectors = []
    for i in range(len(points) - 2):
        p1 = (points[i])
        p2 = (points[i + 1])
        p3 = (points[i + 2])

        vector1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        vector2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        vectors.append((vector1, vector2))
    """
    degrees = []
    for couple in vectors:
        first = couple[0]
        second = couple[1]

        dot = np.dot(first, second)
        ang = np.degrees(np.arccos(dot/(np.linalg.norm(first)*np.linalg.norm(second))))
        degrees.append(ang)

    degrees = np.array(degrees)
    """

    dir2 = dir[1:]
    dir1 = dir[:-1]
    radians = np.arccos(
    (dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

    degrees = np.degrees(radians)
    

    # Quadrant check
    is_bounce = []
    for i in range(0, len(vectors)):
        # Vectors
        first = np.array(vectors[i][0])
        second = np.array(vectors[i][1])
        # is_bounce.append(True)

        v1_x_sign = 1 if first[0] > 0 else -1
        v1_y_sign = 1 if first[1] > 0 else -1
        v2_x_sign = 1 if second[0] > 0 else -1
        v2_y_sign = 1 if second[1] > 0 else -1
            
        bounce = True
            
        # Valleys are parabola peaks because of inverted screen 
        if (v1_y_sign == v2_y_sign and v1_y_sign == 1 and v1_x_sign != v2_x_sign):
            delta_theta = np.degrees(np.arccos(np.dot(first, second)/(np.linalg.norm(first)*np.linalg.norm(second))))
            print(delta_theta)
            if np.abs(delta_theta) < 95 and np.abs(delta_theta) > 20 and (np.linalg.norm(first) > 100 or np.linalg.norm(second) > 100):
                is_bounce.append(True)
                print("True")
            else:
                is_bounce.append(False)
                print("False")
        elif (v1_y_sign == v2_y_sign and v1_y_sign == -1 and v1_x_sign != v2_x_sign):
            is_bounce.append(True) 
        else:
            ang1 = np.arctan(first[1]/first[0])
            ang2 = np.arctan(second[1]/second[0]) 
            # Correct
            if v1_x_sign == 1 and v1_y_sign == 1 and v2_x_sign == -1 and v2_y_sign == -1 : # First vector in first quadrant, second vector in third quadrant
                if ang2 < ang1:
                    is_bounce.append(False)
                else:
                    is_bounce.append(True)
            elif v1_x_sign == -1 and v1_y_sign == 1 and v2_x_sign == 1 and v2_y_sign == -1: # First vector in second quadrant, second vector in fourth quadrant
                if np.abs(ang2) < np.abs(ang1):
                    is_bounce.append(False)
                else:
                    is_bounce.append(True)
            # Correct
            elif v1_x_sign == -1 and v1_y_sign == -1 and v2_x_sign == 1 and v2_y_sign == 1: # First vector in third quadrant, second vector in first quadrant
                if ang2 > ang1:
                    is_bounce.append(False)
                else:
                    is_bounce.append(True)
            # Correct with < sign
            elif v1_x_sign == 1 and v1_y_sign == -1 and v2_x_sign == -1 and v2_y_sign == 1: # First vector in fourth quadrant, second vector in second quadrant
                if np.abs(ang1) < np.abs(ang2):
                    is_bounce.append(False)
                else:
                    is_bounce.append(True)
            else: # If both vectors on the same side then it is 99% a player hitting the ball
                is_bounce.append(True)

        print("Pair: ", first, second, "Bounce: ", bounce, ", Pair number: ", i)
            
    return degrees, is_bounce

def rdp_algo(x, y, args, tolerance=5):
    """With Ramer-Douglas-Pecker algorithm"""

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
        
        def cluster_by_space(groups, x, y, max_distance):
            new_groups = []
            for group in groups:
                points = []
                new_group = [group[0]] # a new group that has as a first item the first element of the current group, which is an index

                for i in group: # all points in the current group
                    points.append(np.array(x[i], y[i]))

                for i in range(len(points)-1):
                    gap = np.linalg.norm(points[i+1]-points[i]) # gap between 2 points in a group
                    if gap > max_distance: # if over a certain threshold
                        new_groups.append(new_group) # I need to create a new group 
                        new_group = [group[i+1]] 
                    else:
                        new_group.append(group[i+1])
                
                new_groups.append(new_group)
            return new_groups
        
        def get_midpoint(groups):
            mps = []
            for group in groups:
                i = 1-int(np.ceil(len(group)/2)) # Group centroid
                mps.append(group[i])
            
            return mps

        groups = cluster_by_time(ix, 1)
        #groups = cluster_by_space(groups, x, y, 15)
        new_indices = get_midpoint(groups)

        
        return new_indices

    min_angle = 25 # min angle = 25 works fine, the smallest it is the better

    points = list(zip(x, y))

    # Use the Ramer-Douglas-Peucker algorithm to simplify the path
    # http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
    # Python implementation: https://github.com/sebleier/RDP/
    simplified = np.array(rdp.rdp(points, tolerance))

    sx, sy = simplified.T    
    
    # compute the direction vectors on the simplified curve
    directions = np.diff(simplified, axis=0)
    
    # Determine the quadrants each vector lies in
    # quadrants_directions = np.sign(directions)

    # Check if the angle is predominantly in the first or fourth quadrant
    theta, is_bounce = angle(directions, simplified)
    # Select the index of the points (in the simplified trajectory) with the greatest theta
    # Large theta is associated with greatest change in direction.
    idx_simple_trajectory = np.where(theta>min_angle)[0]+1

    idx_filtered = []
    for index in idx_simple_trajectory:
        if is_bounce[index-1] == True:
            idx_filtered.append(index)

    # Return real indices of bouncing points
    ix = find_indices(points, list(zip(sx, sy)), idx_filtered)

    # Filter redundant points via clustering
    ix = eliminate_redundant_points(x, y, sx, sy, ix)
    
    return x,y, ix, sx, sy

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

    plt.plot(x, y, '-o')

    if ix.size > 0:
        plt.plot(x[ix+1], y[ix+1], 'rx', markersize=10, label='Bounces')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory with bounces')
    plt.grid(True)
    plt.show()

    draw_video(ix, args)

if __name__ == '__main__':

    # Parse input
    args = parse()

    # Obtain coordinates from csv
    columns = ['x', 'y']
    df = pd.read_csv(args.path_to_csv, usecols=columns)

    """
    # Correcting trajectory
    x = df.iloc[105:160, [0]].values.flatten()
    y = df.iloc[105:160, [1]].values.flatten()
    for i in range(len(x)-1):
        if x[i] == x[i+1]:
            x[i+1] += 1e-5
        if y[i] == y[i+1]:
            y[i+1] += 1e-5

    tck, u = splprep([np.array(x), np.array(y)], s=200)
    new_points = splev(np.linspace(0, 1, 160-105), tck)
    """
    x = df.x
    y = df.y
    """
    print(len(new_points[0]))
    for j in range(len(new_points)):
        x[j+105] = new_points[0][j]
        y[j+105] = new_points[1][j]
        
        #print(x[i], y[i])
    """
    threshold = 0.5

    dx = np.diff(x)
    dy = np.diff(y)
    
    # Find indices where differences exceed the threshold
    outlier_indices = np.where((np.abs(dx) > threshold) | (np.abs(dy) > threshold))[0]

    # Interpolate to smooth out the trajectory around outliers
    """
    smoothed_x = x.copy()
    smoothed_y = y.copy()
    for out in outlier_indices:
        smoothed_x[out] = (smoothed_x[out - 1] + smoothed_x[out + 1]) / 2
        smoothed_y[out] = (smoothed_y[out - 1] + smoothed_y[out + 1]) / 2
    """  
    #rdp_algo(smoothed_x, smoothed_y, args)
    x_5, y_5, ix_5, sx_5, sy_5=rdp_algo(x, y, args, tolerance=15) # tolerance = 15 seems a good compromise, the lower the better
    # x_3, y_3, ix_3, sx_3, sy_3=rdp_algo(x, y, args, tolerance=7)
    #print(ix_5)
    #print(ix_3)
    #final_intercept, original_intercept = find_intercept(ix_5, ix_3, x)
    #print(final_intercept)
    #print(original_intercept)
    plot_bounces(x, y, sx_5, sy_5, ix_5)
    
    draw_video(ix_5, args)