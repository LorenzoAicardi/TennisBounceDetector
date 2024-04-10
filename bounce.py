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

def draw_cross(frame, x, y, size=10, color=(255, 0, 0), thickness=1):
    if not math.isnan(x):
        x, y = int(x), int(y)  # Convert coordinates to integers
        cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
        cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness)

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_csv', type=str, help='path to csv')
parser.add_argument('--path_to_video', type=str, help='path to inferred video')
parser.add_argument('--path_to_output_video', type=str, help='path to save the video with marked bounces')
args = parser.parse_args()


columns = ['timestamp', 'x', 'y']
df = pd.read_csv(args.path_to_csv, usecols=columns)

x = df.x
y = df.y

t = np.degrees(np.arctan2(np.diff(y), np.diff(x)))
dt = np.degrees(np.diff(t))
dt = (dt + 180) % 360 - 180  # Wrap angles to [-180, 180)

# Find indices where angle difference > 150
ix = np.where(np.abs(dt) > 150)[0]

"""Plotting bounces"""

plt.plot(x[ix + 1], y[ix + 1], 'or')  # Plotting points where angle difference > 150
plt.plot(x, y)  
for i in range(len(ix)):
    plt.text(x[ix[i] + 1], y[ix[i] + 1], str(dt[ix[i]]))
plt.show()

"""Drawing on video"""

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
