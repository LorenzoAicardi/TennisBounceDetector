# TennisBounceDetector

Command for inference: `python3 infer_on_video.py --model_path='path_to_repo/TennisBounceDetector/models/tracknet.pt' --video_path='path_to_repo/TennisBounceDetector/videoin/test_tracknet.mp4' --video_out_path='path_to_repo/TennisBounceDetector/videoout/output_tracknet.mp4' --extrapolation`

Command to draw bouncing points on video: `python3 bounce.py --path_to_csv='outcsv/g3c1.csv' --path_to_video='videoout/g3c1out.mp4' --path_to_output_video='g3c1predict.mp4'`

The model and the code for TrackNet was taken from the following repository: https://github.com/yastrebksv/TrackNet
