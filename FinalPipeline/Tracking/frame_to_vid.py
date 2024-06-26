import cv2
import os

image_folder = r'C:/Users/loren/Desktop/tennis/game6/Clip4'
video_name = 'C:/Users/loren/Desktop/TennisBounceDetector/videoin/g6c4.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
