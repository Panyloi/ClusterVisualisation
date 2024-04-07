import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import interpolate
from copy import deepcopy
from random import random

from concave_hull import concave_hull, concave_hull_indexes

def get_cmap(n, name='hsv'):
    cols = (plt.cm.get_cmap(name)
            (np.linspace(0, 1.0, n))[:, :3][:,::-1])
    return cols

def interpolate_points(points, num=100):
    if len(points) < 3:
        return points
    u3=np.linspace(0,1,num,endpoint=True)
    # new_points = splev(u3,tck)

    np_points = np.array(points)
    x = np_points[:, 0]
    y = np_points[:, 1]

    tck,_ = interpolate.splprep([x, y], k=3, s=0)
    new_points = interpolate.splev(u3, tck)
    return list(zip(new_points[0], new_points[1]))

def convert_points(points, flag_radious, num_of_points = 20):
    points_copy = deepcopy(points)

    # only for points
    # for point in points_copy:
    #     points.append((point[0], point[1] + flag_radious))
    #     points.append((point[0], point[1] - flag_radious))
    #     points.append((point[0] - flag_radious, point[1]))
    #     points.append((point[0] + flag_radious, point[1]))

    # circle
    angles = np.linspace(0, 2*np.pi, num_of_points, endpoint=False)

    for point in points_copy:
        for angle in angles:   
            x = point[0] + flag_radious * np.cos(angle)
            y = point[1] + flag_radious * np.sin(angle)
            points.append((x, y))

    return np.array(points)

def visualizate_points(categorized_names, gruped_points, ax, with_iterpolation = True):
    # get random cmap for drawning hulls
    cmap = get_cmap(len(categorized_names))
    np.random.shuffle(cmap)

    for i in range(len(categorized_names)):
        # reshape points into correct shape
        zipped_points = np.hstack((gruped_points[categorized_names[i]]["x"].reshape(-1,1),
                                    gruped_points[categorized_names[i]]["y"].reshape(-1,1)))
        
        # zipped_points = convert_points(list(zipped_points), 20, num_of_points=4)
        # zipped_points = convert_points(list(zipped_points), 10**6)
        # zipped_points = convert_points(list(zipped_points), 0.01)

        # compute concave hull
        # change :param concavity: to add more or less points to concave hull
        idxes = concave_hull_indexes(
            zipped_points[:, :2],
            concavity=2,
            length_threshold=0,
        )

        _idx_points = [zipped_points[idxes[i]] for i in range(len(idxes))]

        # NEW
        # zipped_points = convert_points(_idx_points, 20, num_of_points=10)
        zipped_points = convert_points(_idx_points, 10**6)
        # zipped_points = convert_points(_idx_points, 0.02)

        idxes = concave_hull_indexes(
            zipped_points[:, :2],
            concavity=2,
            length_threshold=20,
        )

        _idx_points = [zipped_points[idxes[i]] for i in range(len(idxes))]
        # END_NEW


        new_points = interpolate_points(_idx_points, 10)

        if with_iterpolation:
            line_to_plot = [
                (new_points[j % len(new_points)], new_points[(j + 1) % len(new_points)])
                for j in range(len(new_points))
            ]
        else:      
            line_to_plot = [(zipped_points[idxes[j]], zipped_points[idxes[(j+1)%len(idxes)]])
                            for j in range(len(idxes))]

        line_segments = LineCollection(segments = np.array(line_to_plot), colors = cmap[i])
        ax.add_collection(line_segments)

def main():

    # prepare given points
    # df = pd.read_csv("./points/kk_swap_2d.csv", sep=';')
    df = pd.read_csv("./points/kamada_l1-mutual_attraction_2d.csv", sep=';')
    # df = pd.read_csv("./points/prefsynth_dataset/coordinates/mds_dapl2_2d.csv", sep=';')

    df["instance_id"] = df["instance_id"].apply(lambda x: x.split(sep="_")[0])

    x_points = df["x"].to_numpy()
    y_points = df["y"].to_numpy()


    categorized_names = df["instance_id"].unique()
    gruped_points = {}

    for i in range(df["instance_id"].nunique()):
        selected_points = df.loc[df["instance_id"] == categorized_names[i]]
        gruped_points[categorized_names[i]] = {"x": selected_points["x"].to_numpy(),
                                               "y": selected_points["y"].to_numpy()}

    fig, ax = plt.subplots()
    ax.scatter(x_points, y_points, s=20)
    ax.set_title('Simple plot')

    visualizate_points(categorized_names, gruped_points, ax, with_iterpolation=False)
    
    plt.show()


if __name__ == "__main__":
    main()