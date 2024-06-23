# TennisBounceDetector

Experimental repository of the IACV project 2023-2024 curated by Lorenzo Aicardi and Gloria Desideri. The project finds the bounces of a tennis ball from a single view video and tries to find their 3D coordinates.

To run the project take a video and place it in the `vin` folder naming it `input.mp4`. Delete the files in `csvout` folder, the `points.npy` file and `camera_matrix.npy` file.

Later this process could be automated.

finally run `python FinalPipeline/main.py`

The model and the code for TrackNet was taken from the following repository: https://github.com/yastrebksv/TrackNet

Court and player tracking from : https://github.com/ArtLabss/tennis-tracking
