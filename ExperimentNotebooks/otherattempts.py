import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_csv', type=str, help='path to csv')
parser.add_argument('--path_to_outvideo', type=str, help='path to csv')
args = parser.parse_args()


columns = ['timestamp', 'x', 'y']
df = pd.read_csv(args.path_to_csv, usecols=columns)

x = df.x
y = df.y

"""Trajectory based approach"""

def calculate_angle(pt1, pt2, pt3):
    #Calculate angle between three points.
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def detect_sudden_changes(points, threshold_angle):
    # Detect sudden changes in direction in a sequence of points.
    sudden_changes = []
    for i in range(1, len(points) - 1):
        angle = calculate_angle(points[i - 1], points[i], points[i + 1])
        if angle > threshold_angle:
            sudden_changes.append(i)
    return sudden_changes

# Example trajectory points
trajectory_points = list(zip(x, y))
print(trajectory_points)

# Define threshold angle (in degrees)
threshold_angle = 160

# Detect sudden changes in direction
ix = detect_sudden_changes(trajectory_points, threshold_angle)

"""Velocity based approach"""

# Function to calculate velocity
def calculate_velocity(x, y, dt):
    dx = np.diff(x)
    dy = np.diff(y)
    distance = np.sqrt(dx**2 + dy**2)
    velocity = distance / dt
    return velocity

time = df.timestamp  # Time points

# Calculate time differences
dt = np.diff(time)

# Calculate velocity
velocity = calculate_velocity(x, y, dt)

# Detect abrupt changes (threshold for velocity change)
threshold = 1.5  # Adjust as needed
ix = np.where(np.abs(np.diff(velocity)) > threshold)[0]

# Plot trajectory
plt.plot(x, y, '-o')

# Mark abrupt changes
if ix.size > 0:
    plt.plot(x[ix+1], y[ix+1], 'rx', markersize=10, label='Abrupt change')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory with Abrupt Changes')
plt.grid(True)
plt.show()
