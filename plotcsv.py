import matplotlib.pyplot as plt 
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_csv', type=str, help='path to csv')
args = parser.parse_args()

columns = ['timestamp', 'x', 'y']
df = pd.read_csv(args.path_to_csv, usecols=columns)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

ax.scatter(df.timestamp, df.x, df.y)
ax.set_xlabel('timestamp')
ax.set_ylabel('x')
ax.set_zlabel('y')
# plt.plot(df.x, df.y)
plt.show()
