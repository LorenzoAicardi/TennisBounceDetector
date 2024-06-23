import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
import pathlib
from pathlib import Path
import rdp
import plotly.graph_objects as go
import plotly.offline as pyo

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
def draw_video(ix,x,y, path_to_video, path_to_output_video):
    if os.path.exists(path_to_video):
        video_capture = cv2.VideoCapture(path_to_video)
    else:
        print("The specified file does not exist.")

    # Video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_to_output_video, fourcc, fps, (width, height))

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

def angle(dir, points):

    """Finds angles between lines in the simplified trajectory."""

    dir2 = dir[1:]
    dir1 = dir[:-1]
    radians = np.arccos(
    (dir1*dir2).sum(axis=1)/(np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

    degrees = np.degrees(radians)
    
    # Vectors for quandrant check
    vectors = []
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
            if np.abs(delta_theta) < 95 and np.abs(delta_theta) > 20 and (np.linalg.norm(first) > 70 or np.linalg.norm(second) > 70):
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
            elif v1_x_sign == -1 and v1_y_sign == -1 and v2_x_sign == 1 and v2_y_sign == 1: # First vector in third quadrant, second vector in first quadrant
                if ang2 > ang1:
                    is_bounce.append(False)
                else:
                    is_bounce.append(True)
            elif v1_x_sign == 1 and v1_y_sign == -1 and v2_x_sign == -1 and v2_y_sign == 1: # First vector in fourth quadrant, second vector in second quadrant
                if np.abs(ang1) < np.abs(ang2):
                    is_bounce.append(False)
                else:
                    is_bounce.append(True)
            else: # If both vectors on the same side then it is 99% a player hitting the ball
                is_bounce.append(True)

        print("Pair: ", first, second, "Bounce: ", bounce, ", Pair number: ", i)
            
    return degrees, is_bounce

def rdp_algo(x, y, tolerance=5):
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

    # Check if the angle is predominantly in the first or fourth quadrant, and if each angle corresponds to a bounce
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

def plot_bounces_plotly(x, y, sx, sy, ix):
    fig = go.Figure()

    # Add the simplified path as a dashed green line
    fig.add_trace(go.Scatter(x=sx, y=sy, mode='lines', line=dict(color='green', dash='dash'), name='simplified path'))

    # Add the bounces as red markers
    fig.add_trace(go.Scatter(x=[x[i] for i in ix], y=[y[i] for i in ix], mode='markers', marker=dict(color='red', size=10), name='bounces'))

    # Add text annotations for each bounce
    for i, idx in enumerate(ix):
        fig.add_annotation(x=x[idx], y=y[idx], text=str(idx), showarrow=True, arrowhead=2, ax=0, ay=-10, font=dict(size=8, color='black'))

    # Invert the y-axis
    fig.update_layout(yaxis=dict(autorange='reversed'))

    # Add legend
    fig.update_layout(legend=dict(x=0, y=1, traceorder='normal'))

    # Show the plot
    fig.show()
    
def detect_bounces(trajectory, out_path, path_to_video, path_to_output_video):
    x = trajectory.x
    y = trajectory.y
    threshold = 0.5

    dx = np.diff(x)
    dy = np.diff(y)
    
    # Find indices where differences exceed the threshold
    outlier_indices = np.where((np.abs(dx) > threshold) | (np.abs(dy) > threshold))[0]
    x_5, y_5, ix_5, sx_5, sy_5=rdp_algo(x, y, tolerance=15)
    bounce_df = pd.DataFrame({'x': x, 'y': y, 'bounce': [1 if i in ix_5 else 0 for i in range(len(x))]})
    bounce_df.to_csv(out_path, index=False)
    
    plot_bounces_plotly(x, y, sx_5, sy_5, ix_5)
    draw_video(ix_5,x,y, path_to_video, path_to_output_video)
    return bounce_df, ix_5, x, y

if __name__ == '__main__':

    # Parse input
    args = parse()

    # Obtain coordinates from csv
    columns = ['x', 'y']
    df = pd.read_csv(args.path_to_csv, usecols=columns)

    x = df.x
    y = df.y
    threshold = 0.5

    dx = np.diff(x)
    dy = np.diff(y)
    
    # Find indices where differences exceed the threshold
    outlier_indices = np.where((np.abs(dx) > threshold) | (np.abs(dy) > threshold))[0]

    x_5, y_5, ix_5, sx_5, sy_5=rdp_algo(x, y, args, tolerance=3) # tolerance = 15 seems a good compromise, the lower the better
    bounce_df = pd.DataFrame({'x': x, 'y': y, 'bounce': [1 if i in ix_5 else 0 for i in range(len(x))]})
    bounce_df.to_csv('bounce.csv', index=False)
    
    plot_bounces(x, y, sx_5, sy_5, ix_5)

    draw_video(ix_5, x, y)