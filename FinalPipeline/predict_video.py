import argparse
import queue
import pandas as pd 
import pickle
import imutils
import os
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import torch
import sys
import time

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from court_detector import CourtDetector

from utils import get_video_properties, get_dtype
from detection import *
from pickle import load


# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--minimap", type=int, default=0)
parser.add_argument("--coordsfile", type=str, default="outcsv.csv")


args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path
minimap = args.minimap
coordsfile = args.coordsfile

if output_video_path == "":
    # output video in same path
    output_video_path = input_video_path.split('.')[0] + "VideoOutput/video_output.mp4"

# get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
print('fps : {}'.format(fps))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# try to determine the total number of frames in the video file
if imutils.is_cv2() is True :
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
else : 
    prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))

# start from first frame
currentFrame = 0

# width and height in TrackNet
width, height = 640, 360
img, img1, img2 = None, None, None



# In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
q = queue.deque()
for i in range(0, 8):
    q.appendleft(None)

# save prediction images as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))



# court
court_detector = CourtDetector()

# players tracker
dtype = get_dtype()
detection_model = DetectionModel(dtype=dtype)

# get videos properties
fps, length, v_width, v_height = get_video_properties(video)
# coords are the trajectory we found with tracknet
coords = pd.read_csv(coordsfile)
frame_i = 0
frames = []
t = []

while True:
  ret, frame = video.read()
  frame_i += 1

  if ret:
    if frame_i == 1:
      print('Detecting the court and the players...')
      lines = court_detector.detect(frame)
    else: # then track it
      lines = court_detector.track_court(frame)
    # COMMENTED FOR NOW
    #detection_model.detect_player_1(frame, court_detector)
    #detection_model.detect_top_persons(frame, court_detector, frame_i)
    
    for i in range(0, len(lines), 4):
      x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
      cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
    new_frame = cv2.resize(frame, (v_width, v_height))
    frames.append(new_frame)
  else:
    break
video.release()
print('Finished!')


# COMMENTED FOR NOW
#detection_model.find_player_2_box()

player_1_val = pd.read_csv('player1_boxes.csv')
player_1_val= player_1_val.values
player_2_val = pd.read_csv('player2_boxes.csv')
player_2_val= player_2_val.values


# second part 

p2_box = np.array([np.array([x1, y1, x2, y2]) for x1, y1, x2, y2 in player_2_val])
# array on None where you find NaN
p2_box = np.where(np.isnan(p2_box), None, p2_box)

detection_model.player_1_boxes = player_1_val
detection_model.player_2_boxes = p2_box


player1_boxes = detection_model.player_1_boxes
player2_boxes = detection_model.player_2_boxes

# save in csv file the boxes
df = pd.DataFrame(player1_boxes)
df.to_csv('player1_boxes.csv', index=False)

df = pd.DataFrame(player2_boxes)
df.to_csv('player2_boxes.csv', index=False)

video = cv2.VideoCapture(input_video_path)
frame_i = 0

last = time.time() # start counting 
# while (True):
for img in frames:
    # print('Tracking the ball: {}'.format(round( (currentFrame / total) * 100, 2)))
    frame_i += 1

    # # detect the ball
    # # img is the frame that TrackNet will predict the position
    # # since we need to change the size and type of img, copy it to output_img
    output_img = img

    output_img = mark_player_box(output_img, player1_boxes, currentFrame-1)
    output_img = mark_player_box(output_img, player2_boxes, currentFrame-1)
    
    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)


    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()

if minimap == 1:
  game_video = cv2.VideoCapture(output_video_path)

  fps1 = int(game_video.get(cv2.CAP_PROP_FPS))

  output_width = int(game_video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(game_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print('game ', fps1)
  output_video = cv2.VideoWriter('VideoOutput/video_with_map.mp4', fourcc, fps, (output_width, output_height))
  
  print('Adding the mini-map...')

  # Remove Outliers 
  x, y = diff_xy(coords[['x', 'y']].values)
  coords=remove_outliers(x, y, coords)
  # Interpolation
  coords = interpolation(coords)
  create_top_view(court_detector, detection_model, coords, fps)
  minimap_video = cv2.VideoCapture('VideoOutput/minimap.mp4')
  fps2 = int(minimap_video.get(cv2.CAP_PROP_FPS))
  print('minimap ', fps2)
  while True:
    ret, frame = game_video.read()
    ret2, img = minimap_video.read()
    if ret:
      output = merge(frame, img)
      output_video.write(output)
    else:
      break
  game_video.release()
  minimap_video.release()

output_video.release()


