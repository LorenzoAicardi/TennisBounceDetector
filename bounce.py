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

def draw_cross(frame, x, y, size=10, color=(255, 0, 0), thickness=2):
    if not math.isnan(x):
        x, y = int(x), int(y)  # Convert coordinates to integers
        cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
        cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str, help='path to csv')
    parser.add_argument('--path_to_video', type=str, help='path to inferred video')
    parser.add_argument('--path_to_output_video', type=str, help='path to save the video with marked bounces')
    return parser.parse_args()

"""With simple angle checking"""
def bounce_angle_wrapping(x, y, args):
    t = np.degrees(np.arctan2(np.diff(y), np.diff(x)))
    dt = np.degrees(np.diff(t))
    dt = (dt + 180) % 360 - 180  # Wrap angles to [-180, 180)

    # Find indices where angle difference > 150
    ix = np.where(np.abs(dt) > 150)[0]

    plot_bounces(x, y, ix, dt)

    draw_video(ix, args)

"""With Ramer-Douglas-Pecker algorithm"""
def rdp_algo(x, y, args):

    def find_indices(points, sp, idx):
        """Find the indices of the turning points found relative to the points list."""
        ix = []

        for index in idx:
            ix.append(points.index(sp[index]))

        return ix

    def angle(dir):
        """
        Returns the angles between vectors.

        Parameters:
        dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

        The return value is a 1D-array of values of shape (N-1,), with each value
        between 0 and pi.

        0 implies the vectors point in the same direction
        pi/2 implies the vectors are orthogonal
        pi implies the vectors point in opposite directions
        """
        dir2 = dir[1:]
        dir1 = dir[:-1]
        return np.arccos((dir1*dir2).sum(axis=1)/(
            np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

    tolerance = 70
    min_angle = np.pi*0.22
    
    points = list(zip(x.to_list(), y.to_list()))
    print(len(points))

    # Use the Ramer-Douglas-Peucker algorithm to simplify the path
    # http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
    # Python implementation: https://github.com/sebleier/RDP/
    simplified = np.array(rdp.rdp(points, tolerance))

    print(len(simplified))
    sx, sy = simplified.T

    print(sx, sy)

    # compute the direction vectors on the simplified curve
    directions = np.diff(simplified, axis=0)
    theta = angle(directions)
    # Select the index of the points with the greatest theta
    # Large theta is associated with greatest change in direction.
    idx = np.where(theta>min_angle)[0]+1

    fig = plt.figure()
    ax =fig.add_subplot(111)

    ax.plot(x, y, 'b-', label='original path')
    ax.plot(sx, sy, 'g--', label='simplified path')
    ax.plot(sx[idx], sy[idx], 'ro', markersize = 10, label='turning points')
    ax.invert_yaxis()
    plt.legend(loc='best')
    plt.show()

    ix = find_indices(points, list(zip(sx, sy)), idx)

    draw_video(ix, args)

"""Plotting bounces"""
def plot_bounces(x, y, ix, dt):
    plt.plot(x[ix + 1], y[ix + 1], 'or')  # Plotting points where angle difference > 150
    plt.plot(x, y)  
    for i in range(len(ix)):
        plt.text(x[ix[i] + 1], y[ix[i] + 1], str(dt[ix[i]]))
    plt.show()

"""Drawing on video"""
def draw_video(ix, args):
    video_capture = cv2.VideoCapture(args.path_to_video)

    # Video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compensation for 1920x1080 videos
    scale = 1
    if width == 1920:
        scale = 2/3

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
            draw_cross(frame, x[item]*scale, y[item]*scale)

        out.write(frame)
        i = i+1

    video_capture.release()
    out.release()

if __name__ == '__main__':
    args = parse()

    columns = ['timestamp', 'x', 'y']
    df = pd.read_csv(args.path_to_csv, usecols=columns)
    x = df.x
    y = df.y

    # bounce_angle_wrapping(x, y, args)
    
    rdp_algo(x, y, args)
