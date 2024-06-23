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

class TennisVideoProcessor:
    def __init__(self, input_video_path, output_video_path="", minimap=0, coordsfile="outcsv.csv"):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path if output_video_path else input_video_path.split('.')[0] + "vout/video_output.mp4"
        self.minimap = minimap
        self.coordsfile = coordsfile

        self.video = cv2.VideoCapture(input_video_path)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        print('fps : {}'.format(self.fps))
        self.output_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.output_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.width, self.height = 640, 360

        self.frame_queue = queue.deque([None] * 8)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = cv2.VideoWriter(self.output_video_path, self.fourcc, self.fps, (self.output_width, self.output_height))

        self.court_detector = CourtDetector()
        self.dtype = get_dtype()
        self.detection_model = DetectionModel(dtype=self.dtype)

        self.fps, self.length, self.v_width, self.v_height = get_video_properties(self.video)
        self.coords = pd.read_csv(self.coordsfile)
        self.lines= None
        self.frames = []
        self._load_player_boxes()
       

    def _load_player_boxes(self):
        if os.path.exists('FinalPipeline/csvout/player1_boxes.csv') and os.path.exists('FinalPipeline/csvout/player2_boxes.csv'):
            self.player_1_boxes = pd.read_csv('FinalPipeline/csvout/player1_boxes.csv').values
            player_2_val = pd.read_csv('FinalPipeline/csvout/player2_boxes.csv').values
            self.player_2_boxes = np.where(np.isnan(player_2_val), None, player_2_val)
            self.detection_model.player_1_boxes = self.player_1_boxes
            self.detection_model.player_2_boxes = self.player_2_boxes
        else:
            self.player_1_boxes = None
            self.player_2_boxes = None
    
        

    def detect_players(self):
        frame_i = 0
        lines_arr = []
        while True:
            ret, frame = self.video.read()
            frame_i += 1
            print(frame_i)

            if not ret:
                break

            if frame_i == 1:
                print('Detecting the court and the players...')
                lines = self.court_detector.detect(frame)
            else:
                lines = self.court_detector.track_court(frame)
            lines_arr.append(lines)
            
            if self.player_1_boxes is None or self.player_2_boxes is None:
                # Perform player detection if boxes are not available
                self.detection_model.detect_player_1(frame, self.court_detector)
                self.detection_model.detect_top_persons(frame, self.court_detector, frame_i)

            for i in range(0, len(lines), 4):
                
                x1, y1, x2, y2 = lines[i], lines[i+1], lines[i+2], lines[i+3]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            new_frame = cv2.resize(frame, (self.v_width, self.v_height))
            self.frames.append(new_frame)

        self.video.release()
        print('Finished!')
        self.detection_model.find_player_2_box()
        
        self.lines = lines_arr
        
        if self.player_1_boxes is None or self.player_2_boxes is None:
            self._save_player_boxes()
        self._save_mid_output()
    
    def track_court(self):
        frame_i = 0
        lines_arr = []
        while True:
            ret, frame = self.video.read()
            frame_i += 1

            if not ret:
                break

            if frame_i == 1:
                print('Detecting the court')
                lines = self.court_detector.detect(frame)
            else:
                lines = self.court_detector.track_court(frame)
            lines_arr.append(lines) 

        self.lines = lines_arr
        self.video.release()
        print('Finished!')
    
    
    def draw_court_and_players(self, input_video_path):
        game_video = cv2.VideoCapture(input_video_path)
        fps1 = int(game_video.get(cv2.CAP_PROP_FPS))
        output_width = int(game_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(game_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video = cv2.VideoWriter('FinalPipeline/vout/court_and_players.mp4', self.fourcc, fps1, (output_width, output_height))
        print('Adding the court and players...')
        frame_i = 0
        while True:
            ret, frame = game_video.read()
            if ret:
                for i in range(0, len(self.lines[frame_i]), 4):
                    #print(self.lines[frame_i])
                    x1, y1, x2, y2 = self.lines[frame_i][i], self.lines[frame_i][i+1], self.lines[frame_i][i+2], self.lines[frame_i][i+3]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                frame = mark_player_box(frame, self.player_1_boxes, frame_i)
                frame = mark_player_box(frame, self.player_2_boxes, frame_i)
                output_video.write(frame)
            else:
                break
            frame_i += 1

        game_video.release()
        output_video.release()
            

    def _save_player_boxes(self):
        player1_boxes = self.detection_model.player_1_boxes
        player2_boxes = self.detection_model.player_2_boxes
        self.player_1_boxes= player1_boxes
        self.player_2_boxes = player2_boxes

        df = pd.DataFrame(self.player_1_boxes)
        df.to_csv('FinalPipeline/csvout/player1_boxes.csv', index=False)

        df = pd.DataFrame(self.player_2_boxes)
        df.to_csv('FinalPipeline/csvout/player2_boxes.csv', index=False)

    def _save_mid_output(self):
        frame_i = 0
        last = time.time()

        for img in self.frames:
            frame_i += 1
            output_img = img

            if self.player_1_boxes is not None:
                output_img = mark_player_box(output_img, self.player_1_boxes, self.current_frame - 1)
            if self.player_2_boxes is not None:
                output_img = mark_player_box(output_img, self.player_2_boxes, self.current_frame - 1)

            PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray(PIL_image)

            opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            self.output_video.write(opencvImage)

            self.current_frame += 1

        self.output_video.release()

    def draw_minimap(self, input_video_path):
        game_video = cv2.VideoCapture(input_video_path)
        fps1 = int(game_video.get(cv2.CAP_PROP_FPS))
        output_width = int(game_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(game_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video = cv2.VideoWriter('FinalPipeline/vout/final_output.mp4', self.fourcc, fps1, (output_width, output_height))
        print('Adding the mini-map...')
        bounce_var = self.coords['bounce'].values
        x, y = diff_xy(self.coords[['x', 'y']].values)
        self.coords = remove_outliers(x, y, self.coords)
        self.coords = interpolation(self.coords)
        self.coords['bounce'] = bounce_var
        print(self.coords)
        create_top_view(self.court_detector, self.detection_model, self.coords, fps1)

        minimap_video = cv2.VideoCapture('FinalPipeline/vout/minimap.mp4')
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
