import matplotlib.pyplot as plt 
import pandas as pd
import argparse

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
    df = pd.read_csv(args.path_to_csv, usecols=columns)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.scatter(df.x, df.y)
    for i in range(len(df.x)):
        plt.plot(df.x[i:i+2], df.y[i:i+2], 'ro-')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plt.plot(df.x, df.y)
    ax.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str, help='path to csv')
    args = parser.parse_args()

    # view_3d(args)
    view_2d(args)