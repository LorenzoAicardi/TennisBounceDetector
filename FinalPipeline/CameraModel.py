import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.offline as pyo


class CameraModel:
    def __init__(self, video_path, ground_points, E_top,F_top, camera_matrix=None):
        self.video = cv2.VideoCapture(video_path)
        camera_matrix = camera_matrix
        self.ground_points = ground_points
        self.points=None
        self.M=None 
        self.intersection_point=None
        self.E_top = E_top
        self.F_top = F_top
        self.image_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def selct_points_and_calibrate(self):
        # select 4 points on the video to calibrate the camera
        ret, frame = self.video.read()
        cv2.imshow('Select Points', frame)
        print('Select 4 points on the video starting from top left corner, select the two poles tip')
        points = []
        
        def get_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                print("Point selected: ({}, {})".format(x, y))
        cv2.setMouseCallback('Select Points', get_points)

        # Wait for the user to select points
        cv2.waitKey(0)

        # Close the image window
        cv2.destroyAllWindows()

        # Print the selected points
        print("Selected Points:", points)
        self.points=points
        self.points = np.array(points)
        self.points = self.points.astype(np.float32)
        
        self.points[:,1] = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) - self.points[:,1]
        np.save('points.npy', self.points)

    def find_intersection_point(self,p1, p2, p3, p4):
        # Extracting coordinates for line 1
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Direction vector for line 1
        d1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        
        # Parametric values for line 1
        t1 = np.linspace(-1, 2, 100)
        
        # Line 1 points
        x_line1 = x1 + t1 * d1[0]
        y_line1 = y1 + t1 * d1[1]
        z_line1 = z1 + t1 * d1[2]
        
        # Extracting coordinates for line 2
        x3, y3, z3 = p3
        x4, y4, z4 = p4
        
        # Direction vector for line 2
        d2 = np.array([x4 - x3, y4 - y3, z4 - z3])
        
        # Parametric values for line 2
        t2 = np.linspace(-1, 2, 100)
        
        # Line 2 points
        x_line2 = x3 + t2 * d2[0]
        y_line2 = y3 + t2 * d2[1]
        z_line2 = z3 + t2 * d2[2]
        
        # Solve for the intersection point
        A = np.array([
            [d1[0], -d2[0]],
            [d1[1], -d2[1]],
            [d1[2], -d2[2]]
        ])
        B = np.array([x3 - x1, y3 - y1, z3 - z1])
        t = np.linalg.lstsq(A, B, rcond=None)[0]
        
        # Calculate intersection point coordinates
        intersection_x = x1 + t[0] * d1[0]
        intersection_y = y1 + t[0] * d1[1]
        intersection_z = z1 + t[0] * d1[2]
        
        return intersection_x, intersection_y, intersection_z
    def calibrate_camera(self):
        # check if image and ground points are not None
        if self.points is None or self.ground_points is None:
            raise ValueError('Image and ground points are not selected')
        # Convert the points to numpy array
        image_points = self.points[:4]
        ground_points = np.array(self.ground_points)
        # transform in float32
        ground_points = ground_points.astype(np.float32)
        # subtract image height from y
        print('Image points:', image_points)
        M = cv2.getPerspectiveTransform(image_points,ground_points)
        self.M=M
        # save the camera matrix
        np.save('camera_matrix.npy', M)
    
    def load_camera_matrix(self):
        M = np.load('camera_matrix.npy')
        self.M=M
    
    def plot_lines_points_intersection_3d_plotly(self,p1, p2, p3, p4):
        # Extracting coordinates for line 1
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Direction vector for line 1
        d1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        
        # Parametric values for line 1
        t1 = np.linspace(-1, 2, 500)
        
        # Line 1 points
        x_line1 = x1 + t1 * d1[0]
        y_line1 = y1 + t1 * d1[1]
        z_line1 = z1 + t1 * d1[2]
        
        # Extracting coordinates for line 2
        x3, y3, z3 = p3
        x4, y4, z4 = p4
        
        # Direction vector for line 2
        d2 = np.array([x4 - x3, y4 - y3, z4 - z3])
        
        # Parametric values for line 2
        t2 = np.linspace(-1, 2, 500)
        
        # Line 2 points
        x_line2 = x3 + t2 * d2[0]
        y_line2 = y3 + t2 * d2[1]
        z_line2 = z3 + t2 * d2[2]
        
        # Solve for the intersection point
        A = np.array([
            [d1[0], -d2[0]],
            [d1[1], -d2[1]],
            [d1[2], -d2[2]]
        ])
        B = np.array([x3 - x1, y3 - y1, z3 - z1])
        t = np.linalg.lstsq(A, B, rcond=None)[0]
        
        # Calculate intersection point coordinates
        intersection_x = x1 + t[0] * d1[0]
        intersection_y = y1 + t[0] * d1[1]
        intersection_z = z1 + t[0] * d1[2]
        
        # Create trace for line 1
        line_trace1 = go.Scatter3d(
            x=x_line1,
            y=y_line1,
            z=z_line1,
            mode='lines',
            name='Line 1',
            line=dict(color='blue', width=2)
        )
        
        # Create trace for line 2
        line_trace2 = go.Scatter3d(
            x=x_line2,
            y=y_line2,
            z=z_line2,
            mode='lines',
            name='Line 2',
            line=dict(color='red', width=2)
        )
        
        # Create trace for points
        points_trace = go.Scatter3d(
            x=[x1, x2, x3, x4],
            y=[y1, y2, y3, y4],
            z=[z1, z2, z3, z4],
            mode='markers',
            name='Points',
            marker=dict(color='black', size=5)
        )
        
        # Create trace for intersection point
        intersection_trace = go.Scatter3d(
            x=[intersection_x],
            y=[intersection_y],
            z=[intersection_z],
            mode='markers',
            name='Intersection Point',
            marker=dict(color='green', size=5)
        )
        
        # Layout
        layout = go.Layout(
            title='Lines, Points, and Intersection Point in 3D',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Create the figure
        fig = go.Figure(data=[line_trace1, line_trace2, points_trace, intersection_trace], layout=layout)
        
        # Plot the figure
        pyo.iplot(fig)  
        # return intersection point as np array
        return np.array([intersection_x, intersection_y, intersection_z])  
        
    def load_points(self):
        #load points array
        self.points = np.load('points.npy')
        
    def find_camera_center(self):
        if self.M is None:
            raise ValueError('Camera matrix is not found')
        # get last two points of points array
        print(self.points)
        E_image = np.array(self.points[4])
        F_image = np.array(self.points[5])
        
        E_transformed = np.dot(self.M, np.array([E_image[0], E_image[1], 1]))
        E_transformed /= E_transformed[2]

        F_transformed = np.dot(self.M, np.array([F_image[0], F_image[1], 1]))
        F_transformed /= F_transformed[2]
        E_g = E_transformed
        F_g = F_transformed
        E_g[2]= 0
        F_g[2]= 0
        print('E_g:', E_g)
        print('F_g:', F_g)
        print('E_top:', self.E_top)
        print('F_top:', self.F_top)
        self.intersection_point = self.plot_lines_points_intersection_3d_plotly(E_g, self.E_top, F_g, self.F_top)
        
