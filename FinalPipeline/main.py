from Tracking.BallTracker import BallTracker
import pandas as pd
from bounce import detect_bounces
from bounce import draw_video
from TennisVideoProcessor import TennisVideoProcessor
import numpy as np
from CameraModel import CameraModel
import os
import plotly.graph_objects as go
import plotly.offline as pyo

def plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)  # Normalize the normal vector
    return normal

# function to transform in homogeneous coordinates
def homogeneous(x, y, image_size=720):
    return np.array([x, image_size-y, 1])
# function to project on ground plane
def project_on_ground_plane(M, point):
    point = np.dot(M,point)
    point /= point[2]
    return point
def find_box_points(base1, base2, height):
    p1 = base1
    p2 = base2
    p3 = np.array([base1[0], base1[1], height])
    p4 = np.array([base2[0], base2[1], height])
    return p1, p2, p3, p4
def find_plane_points_normal(p1, p2, height):
    box_height = 700
    
    # Define points for the bounding box
    p7 = p1 + np.array([0, 0, box_height])
    p8 = p2 + np.array([0, 0, box_height])
    normal = plane_from_points(p1, p2, p7)
    return p7, p8, normal

def intersect_line_plane(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    Intersects a line defined by two points p0 and p1 with a plane defined by its center point p_co and normal vector p_no.
    
    Parameters:
    - p0, p1: Tuple representing the start and end points of the line.
    - p_co: Tuple representing a point on the plane.
    - p_no: Tuple representing the normal vector of the plane.
    - epsilon: Small value to avoid division by zero errors.
    
    Returns:
    - Intersection point as a tuple or None if no intersection exists.
    """
    u = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
    dot = sum(a*b for a, b in zip(p_no, u))
    
    if abs(dot) > epsilon:
        w = (p0[0]-p_co[0], p0[1]-p_co[1], p0[2]-p_co[2])
        fac = -(sum(a*b for a, b in zip(p_no, w)) / dot)
        u_scaled = (u[0]*fac, u[1]*fac, u[2]*fac)
        return (p0[0]+u_scaled[0], p0[1]+u_scaled[1], p0[2]+u_scaled[2])
    
    return None


if __name__ == '__main__':
    if os.path.exists('FinalPipeline/csvout/output.csv'):
        trajectory = pd.read_csv('FinalPipeline/csvout/output.csv')
    else:
        tracker = BallTracker(model_path='FinalPipeline/Tracking/models/tracknet.pt', extrapolation=True)
    
        trajectory=tracker.track_ball('FinalPipeline/vin/input.mp4', 'FinalPipeline/vout/output.mp4', 'FinalPipeline/csvout/output.csv')
        print(trajectory)
    bounces, ix_5, x,y = detect_bounces(trajectory, 'FinalPipeline/csvout/bounces.csv', path_to_video='FinalPipeline/vout/output.mp4', path_to_output_video='FinalPipeline/vout/output_bounces.mp4')
    #print(bounces)
    processor= TennisVideoProcessor('FinalPipeline/vin/input.mp4', 'FinalPipeline/vout/output_map.mp4', coordsfile='FinalPipeline/csvout/bounces.csv')
    # if csvout boxes do not exist, detect players
    if processor.player_1_boxes is None or processor.player_2_boxes is None:
        processor.detect_players()
    # draw minimap on current output video
    processor.track_court()
    processor.draw_court_and_players(input_video_path='FinalPipeline/vout/output_bounces.mp4')
    processor.draw_minimap(input_video_path='FinalPipeline/vout/court_and_players.mp4')
    
    # calibrate camera
    ground_points = np.array([
                [91,2377], [1188,2377], [1188,0],[91,0]
            ])
    E_top = np.array([0,  1188, 107])
    F_top = np.array([1279,1188, 107])
    camera = CameraModel('FinalPipeline/vin/input.mp4', ground_points, E_top, F_top)
    
    
    # if camera matrix does not exist, select points and calibrate
    if not os.path.exists('camera_matrix.npy'):
        camera.selct_points_and_calibrate()
        camera.calibrate_camera()
    else:
        camera.load_camera_matrix()
        camera.load_points()
    camera.find_camera_center()
    print('Intersection point:', camera.intersection_point)
    camera_center=camera.intersection_point
    print('Camera matrix:', camera.M)
    
    # plot baounces in 3d
    df_bounces= pd.read_csv('FinalPipeline/csvout/bounces.csv')
    player1_boxes= pd.read_csv('FinalPipeline/csvout/player1_boxes.csv')
    player2_boxes= pd.read_csv('FinalPipeline/csvout/player2_boxes.csv')
    
    # geta all bounces
    bounces = df_bounces[df_bounces['bounce'] == 1]
    # get all y coordintes
    y = bounces['y'].values
    y = camera.image_height-y
    x = bounces['x'].values
    
    # fint top left corner of image
    top_left = np.array([0,camera.image_height,1])
    # project on the ground
    top_left_ground = project_on_ground_plane(camera.M, top_left)
    top_left_ground/=top_left_ground[2]
    top_left_ground
    
    
    criteria= 0 # 0 minore 1 maggiore
    #understand if the ball is going up or down
    if y[0] < y[1]:
        criteria = 1
    initial_criteria = criteria
    image_y=720
    fig = go.Figure()
    # y distance E_top and camera center
    y_distance = E_top[1] - camera_center[1]

    center_flipped = np.array([camera_center[0],camera_center[1]+ 2*top_left_ground[1], camera_center[2]]) 
    #fig.add_trace(go.Scatter3d(x=[camera_center[0], center_flipped[0]], y=[camera_center[1], center_flipped[1]], z=[camera_center[2], center_flipped[2]], mode='markers+lines', name='Camera Center'))
    fig.add_trace(go.Scatter3d(x=[E_top[0], F_top[0]], y=[E_top[1], F_top[1]], z=[E_top[2], F_top[2]], mode='markers+lines', name='Original', marker=dict(color='green', size=2)))
    indexes= []
    intersectios_p1= []
    intersections_p2= []
    ground_p = []

    for i in range(len(y)-1):
        if criteria == 1:
            if y[i+1] > y[i]:
                print("Bounce on player at point: ({}, {})".format(x[i], y[i]))
                #plt.scatter(x[i], y[i], c='r')
                # add i label
                #plt.text(x[i], y[i], str(i+1), color='red', fontsize=12)
                
                #transform bounce in homogeneous coordinates
                bounce = homogeneous(x[i], y[i])
                
                # get player box
                player_box = player1_boxes.iloc[i]
                box = player_box[['0', '1', '2', '3']].values
                box = box.reshape(-1, 2)
                box = box.astype(int)
                top_left = box[0]
                bottom_right = box[1]
                bottom_left = (top_left[0], bottom_right[1])
                
                bottom_right = homogeneous(bottom_right[0], bottom_right[1])
                bottom_left = homogeneous(bottom_left[0], bottom_left[1])
                
                # project on the ground plane
                ground_bounce = project_on_ground_plane(camera.M, bounce)
                ground_bottom_right = project_on_ground_plane(camera.M, bottom_right)
                ground_bottom_left = project_on_ground_plane(camera.M, bottom_left)
                
                # z-coordinate 0
                ground_bounce[2]= 0
                ground_bottom_right[2]= 0
                ground_bottom_left[2]= 0
                
                #get plane points and normal
                p7, p8, normal = find_plane_points_normal(ground_bottom_left, ground_bottom_right, 700)
                intersection_point = intersect_line_plane(camera_center, ground_bounce, p8, normal)
                # add to intersections p1
                intersectios_p1.append(intersection_point)
                
                #int_trace = go.Scatter3d(x=[intersection_point[0]], y=[intersection_point[1]], z=[intersection_point[2]], mode='markers', name='Intersection Point', marker=dict(color='red', size=2))
                #fig.add_trace(int_trace)
                
                # add index to list
                indexes.append(i)
                criteria = 0
            else:
                bounce = homogeneous(x[i], y[i])
                ground_bounce = project_on_ground_plane(camera.M, bounce)
                ground_bounce[2]= 0
                ground_p.append(ground_bounce)
                
                
        if criteria == 0:
            if y[i+1] < y[i]:
                print("Bounce on player at point: ({}, {})".format(x[i], y[i]))
                #plt.scatter(x[i], y[i], c='r')
                #plt.text(x[i], y[i], str(i+1), color='red', fontsize=12)
                indexes.append(i)
                
                player_box = player2_boxes.iloc[i]
                # chack if not na
                if player_box.isna().sum() == 0:
                    box = player_box[['0', '1', '2', '3']].values
                    box = box.reshape(-1, 2)
                    box = box.astype(int)
                    top_left = box[0]
                    bottom_right = box[1]
                    bottom_left = (top_left[0], bottom_right[1])
                    
                    bottom_right = homogeneous(bottom_right[0], bottom_right[1])
                    bottom_left = homogeneous(bottom_left[0], bottom_left[1])
                    
                    bounce = homogeneous(x[i], y[i])
                    
                    # project on the ground plane
                    ground_bottom_right = project_on_ground_plane(camera.M, bottom_right)
                    ground_bottom_left = project_on_ground_plane(camera.M, bottom_left)
                    ground_bounce = project_on_ground_plane(camera.M, bounce)
                
                    
                    
                    
                    # z-coordinate 0
                    ground_bottom_right[2]= 0
                    ground_bottom_left[2]= 0
                    ground_bounce[2]= 0
                    
                    #get plane points and normal
                    p7, p8, normal = find_plane_points_normal(ground_bottom_left, ground_bottom_right, 700)
                    intersection_point = intersect_line_plane(center_flipped, ground_bounce, p8, normal)
                    intersections_p2.append(intersection_point)
                    #int_trace = go.Scatter3d(x=[intersection_point[0]], y=[intersection_point[1]], z=[intersection_point[2]], mode='markers', name='Intersection Point', marker=dict(color='blue', size=2))
                    #fig.add_trace(int_trace)
                
                
                criteria = 1
            else:
                bounce = homogeneous(x[i], y[i])
                ground_bounce = project_on_ground_plane(camera.M, bounce)
                ground_bounce[2]= 0
                ground_p.append(ground_bounce)
    
    # scatter intersections p1
    intersections_p1 = np.array(intersectios_p1)
    fig.add_trace(go.Scatter3d(x=intersections_p1[:, 0], y=intersections_p1[:, 1], z=intersections_p1[:, 2], mode='markers', name='Intersections P1', marker=dict(color='red', size=2)))
    #scatter intersections p2
    intersections_p2 = np.array(intersections_p2)
    fig.add_trace(go.Scatter3d(x=intersections_p2[:, 0], y=intersections_p2[:, 1], z=intersections_p2[:, 2], mode='markers', name='Intersections P2', marker=dict(color='blue', size=2)))  
    # scatter ground 
    ground_p = np.array(ground_p)
    fig.add_trace(go.Scatter3d(x=ground_p[:, 0], y=ground_p[:, 1], z=ground_p[:, 2], mode='markers', name='Ground P1', marker=dict(color='green', size=2)))

    # tennis court
    court = np.array([[91,2377,0], [1188,2377,0], [1188,0,0],[91,0,0]])
    # plot tennis court mesh
    court_trace = go.Mesh3d(
        x=[court[0][0], court[1][0], court[2][0], court[3][0], court[0][0]],
        y=[court[0][1], court[1][1], court[2][1], court[3][1], court[0][1]],
        z=[court[0][2], court[1][2], court[2][2], court[3][2], court[0][2]],
        i=[0, 0, 0, 0],
        j=[1, 2, 3, 0],
        k=[2, 3, 1, 1],
        opacity=0.5,
        color='blue'
    )
    fig.add_trace(court_trace)


    fig.show()
    

    
    
    