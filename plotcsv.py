import matplotlib.pyplot as plt 
import pandas as pd
import argparse
import numpy as np
from scipy.interpolate import splprep, splev

def view_3d():
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

def view_2d(args):
    columns = ['x', 'y']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    #x = df.x
    #y = df.y
    
    df = pd.read_csv(args.path_to_csv, usecols=columns)
    x = df.iloc[105:160, [0]].values.flatten()
    y = df.iloc[105:160, [1]].values.flatten()
    for i in range(len(x)-1):
        if x[i] == x[i+1]:
            x[i+1] += 1e-5
        if y[i] == y[i+1]:
            y[i+1] += 1e-5

    tck, u = splprep([np.array(x), np.array(y)], s=0)
    new_points = splev(np.linspace(0, 1, 160-105), tck)

    #print(len(new_points))
    x = new_points[0]
    y = new_points[1]

    ax.scatter(x, y)
    for i in range(len(x)):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str, help='path to csv')
    args = parser.parse_args()

    # view_3d(args)
    view_2d(args)