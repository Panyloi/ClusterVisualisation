import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import interpolate
from copy import deepcopy
from random import random
import time
from sklearn import preprocessing as pr
from collections import deque

from concave_hull import concave_hull, concave_hull_indexes

epsilon = 10e-12


def get_cmap(n, name="hsv"):
    cols = plt.cm.get_cmap(name)(np.linspace(0, 1.0, n))[:, :3][:, ::-1]
    return cols


def interpolate_points(points, num=100):
    if len(points) < 3:
        return points
    u3 = np.linspace(0, 1, num, endpoint=True)
    # new_points = splev(u3,tck)

    np_points = np.array(points)
    x = np_points[:, 0]
    y = np_points[:, 1]

    tck, _ = interpolate.splprep([x, y], k=3, s=0)
    new_points = interpolate.splev(u3, tck)
    return list(zip(new_points[0], new_points[1]))


def _change_list(tab):
    tab[0] = tab[1]
    tab[1] = tab[2]
    tab[2] = 0

def _reset_dict(d):
    for key in d.keys():
        d[key] = 0

def handle_corners_points(points, default_points_dict, num_of_points = 20):
    if len(points) <= 1:
        return points

    num_of_sets = len(points)
    first_points_set = set(points[0])
    after_curve = False
    visited_points = {}

    bourder_value = int(
        num_of_points * 0.3
    )  # 0.3 is hyperparapeter that euqals 30% of all points (TODO: adjust this value in further attmpts)

    new_points = []
    points_queue = deque()
    straight_points_list = [0, 0, 1]  # 1 == straight_points
    test_point = None

    p0 = min(points[0], key=lambda x: (x[1], x[0]))
    points_queue.append(default_points_dict[(p0[0], p0[1])])
    test_point = default_points_dict[
        (p0[0], p0[1])
    ]  # required value for cicuit of main point is grather than 6 !!! (TODO: check this)

    i_array = 0

    # lists of considered points
    current_points_array = points[0]
    next_points_array = points[1]

    # point that is currently handle
    current_point = p0

    # last point considered
    last_point = (
        current_points_array[0]
        if current_points_array[0] != p0
        else current_points_array[1]
    )

    while True:

        # create new set of points
        points_union = current_points_array + next_points_array

        for i in range(len(points_union)):

            if orient(current_point, last_point, points_union[i]) == -1 or (
                orient(current_point, last_point, points_union[i]) == 0
                and distance(last_point, current_point)
                < distance(current_point, points_union[i])
            ):
                if points_union[i] in next_points_array:
                    second_point = points_union[i]
                last_point = points_union[i]

        current_point = last_point
        visited_points[default_points_dict[(current_point[0], current_point[1])]] = 1
        # if second_point in first_points_set:
        if i_array > num_of_sets + 1:
            break

        if current_point in current_points_array:
            straight_points_list[1] += 1
        else:
            _point = default_points_dict[(current_point[0], current_point[1])]
            if len(points_queue) == 3:  # "if" is to handle first start of function
                test_point = points_queue.popleft()

                if (
                    _point == test_point and straight_points_list[1] > bourder_value
                ) or _point != test_point:
                    after_curve = True
                    new_points.append(current_points_array)

                elif _point != test_point and visited_points[_point] == 0:
                    visited_points[_point] += 1
                    new_points.append(current_points_array)
                
                if after_curve:
                    _reset_dict(visited_points)
                    after_curve = False

            current_points_array = next_points_array
            next_points_array = points[i_array % len(points)]
    
            _change_list(straight_points_list)
            straight_points_list[2] += 1
            
            i_array += 1
            points_queue.append(_point)


    return new_points


def create_new_circuit(points, dict_circuit_points, dict_main_points):
    idxes = concave_hull_indexes(
        np.array(points)[:, :2],
        concavity=1,  # TODO: check other numbers for parameter
        length_threshold=0,
    )
    # debbug
    # print(f"{new_set_of_points}\n\n\n")
    _idx_points = [points[idxes[i]] for i in range(len(idxes))]
    line_to_plot = [
        (_idx_points[j % len(_idx_points)], _idx_points[(j + 1) % len(_idx_points)])
        for j in range(len(_idx_points))
    ]
    # print(f"Line to plot\n{line_to_plot}\n\n\n")

    x, y = points[idxes[0]]
    current_point = dict_circuit_points[(x, y)]
    # print(f"IDXES: {len(idxes)}")
    x_current, y_current = current_point
    new_set_of_points = []

    for i in range(len(idxes)):
        x, y = points[idxes[i]]
        if current_point != dict_circuit_points[(x, y)]:
            x_current, y_current = current_point
            new_set_of_points.append(dict_main_points[(x_current, y_current)])
            current_point = dict_circuit_points[(x, y)]

    if len(new_set_of_points) == 0:
        new_set_of_points.append(dict_main_points[(x_current, y_current)])
    # print(f"new_set_of_points\n{new_set_of_points}\n\n\n")
    # print(f"LEN{len(new_set_of_points)}\n\n\n")
    return new_set_of_points


def convert_points(points, flag_radious, num_of_points=20):
    points_copy = deepcopy(points)
    jarvis_points = []
    circuit_points = {}
    main_points_of_circuit = {}

    # circle
    angles = np.linspace(0, 2 * np.pi, num_of_points, endpoint=False)

    for point in points_copy:
        new_set_of_points = []
        for angle in angles:
            x = point[0] + flag_radious * np.cos(angle)
            y = point[1] + flag_radious * np.sin(angle)
            points.append((x, y))
            new_set_of_points.append((x, y))

            circuit_points[(x, y)] = (point[0], point[1])
        main_points_of_circuit[(point[0], point[1])] = new_set_of_points

        # jarvis_points.append(new_set_of_points)
        jarvis_points.extend(new_set_of_points)
    # print(jarvis_points)
    
    new_circuit = create_new_circuit(
        jarvis_points, circuit_points, main_points_of_circuit
    )
    # print(f"new_circuit{new_circuit}\n\n\n")
    print(f"CORNER POINTS{handle_corners_points(new_circuit, circuit_points)}\n\n\n")

    return np.array(points), handle_corners_points(new_circuit, circuit_points)
    # return np.array(points), jarvis_points


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def orient(p, q, r):
    result = (p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0])
    if result > epsilon:
        return 1
    elif result < -epsilon:
        return -1
    else:
        return 0


# similarity to Jarvis algorithms
"""
params:
    points : set of circle points 
             [[(x11, y11), (x21, y21), ..., (xn1, yn1)], [(x12, y12), (x22, y22), ..., (xm2, ym2)], ...]

return:
    points that create outline for cluster points

"""

def greedy_selecting(points):
    # handle 1 point cluster
    # print("WITAM")
    # print(points)
    # print("\n\n\n")
    if len(points) <= 1:
        return points[0]

    num_of_sets = len(points)
    first_points_set = set(points[0])
    second_points_set = set(points[1])
    flag = False
    straight_points = 0

    p0 = min(points[0], key=lambda x: (x[1], x[0]))

    # actual points to return
    outline_points = []

    i_array = 2

    # lists of considered points
    current_points_array = points[0]
    next_points_array = points[1]
    second_next_points_array = points[2]

    # point that is currently handle
    current_point = p0

    second_point = next_points_array[0]

    # last point considered
    last_point = (
        current_points_array[0]
        if current_points_array[0] != p0
        else current_points_array[1]
    )

    while True:

        # create new set of points
        points_union = (
            current_points_array + next_points_array + second_next_points_array
        )

        for i in range(len(points_union)):

            if orient(current_point, last_point, points_union[i]) == -1 or (
                orient(current_point, last_point, points_union[i]) == 0
                and distance(last_point, current_point)
                < distance(current_point, points_union[i])
            ):
                if points_union[i] not in current_points_array:
                    if not flag and i_array >= 3:
                        flag = not flag
                    second_point = points_union[i]
                last_point = points_union[i]

        current_point = last_point

        # if second_point in first_points_set:
        if i_array >= num_of_sets + 2:
            # if flag and i_array == num_of_sets and (second_point in first_points_set or current_point in second_points_set):
            print(i_array)
            break

        if current_point in current_points_array:
            straight_points += 1

        if current_point in next_points_array:
            print(f"straight_points: {straight_points}")
            straight_points = 0

            i_array += 1
            current_points_array = next_points_array
            next_points_array = second_next_points_array
            second_next_points_array = points[i_array % len(points)]
        elif current_point in second_next_points_array:
            print(f"straight_points: {straight_points}")
            straight_points = 0

            i_array += 1
            current_points_array = second_next_points_array
            next_points_array = points[i_array % len(points)]
            i_array += 1
            second_next_points_array = points[i_array % len(points)]

        outline_points.append(current_point)

    return outline_points

def visualizate_points(categorized_names, gruped_points, ax, with_iterpolation=True):
    # get random cmap for drawning hulls
    cmap = get_cmap(len(categorized_names))
    np.random.shuffle(cmap)

    for i in range(len(categorized_names)):
        # reshape points into correct shape
        zipped_points = np.hstack(
            (
                gruped_points[categorized_names[i]]["x"].reshape(-1, 1),
                gruped_points[categorized_names[i]]["y"].reshape(-1, 1),
            )
        )

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

        # zipped_points = convert_points(_idx_points, 20, num_of_points=10)[0]
        zipped_points, to_jarvis = convert_points(_idx_points, 10**6, num_of_points=20)
        # zipped_points = convert_points(_idx_points, 0.02)
        # zipped_points = convert_points(_idx_points, 0.02)[0]

        # start reverse jarvis algorithm

        # NEW

        # outline_points = np.array(greedy_selecting(to_jarvis))
        _idx_points = np.array(greedy_selecting(to_jarvis))
        # print("OUTLINE POINTS")
        # print(outline_points)

        # idxes = concave_hull_indexes(
        #     outline_points[:, :2],
        #     concavity=2,
        #     length_threshold=20,
        # )

        # _idx_points = [outline_points[idxes[i]] for i in range(len(idxes))]

        # END_NEW

        # idxes = concave_hull_indexes(
        #     zipped_points[:, :2],
        #     concavity=2,
        #     length_threshold=20,
        # )

        # _idx_points = [zipped_points[idxes[i]] for i in range(len(idxes))]

        new_points = interpolate_points(_idx_points, 1000)

        if with_iterpolation:
            line_to_plot = [
                (new_points[j % len(new_points)], new_points[(j + 1) % len(new_points)])
                for j in range(len(new_points))
            ]
        else:
            line_to_plot = [
                (
                    _idx_points[j % len(_idx_points)],
                    _idx_points[(j + 1) % len(_idx_points)],
                )
                for j in range(len(_idx_points))
            ]
            # line_to_plot = [(zipped_points[idxes[j]], zipped_points[idxes[(j+1)%len(idxes)]])
            # for j in range(len(idxes))]

        line_segments = LineCollection(segments=np.array(line_to_plot), colors=cmap[i])
        # line_segments = LineCollection(segments = np.array(line_to_plot))
        ax.add_collection(line_segments)


def get_pierre_syntetic(df):
    df_scatter = df.copy()
    df["instance_id"] = df["instance_id"].apply(lambda x: x.split(sep="_")[0])

    x_points = df["x"].to_numpy()
    y_points = df["y"].to_numpy()
    rows = []
    flag = False

    for row in df.iterrows():
        if row[1]["instance_id"] == "syntetic":
            flag = not flag
            continue

        if flag:
            rows.append(pd.Series([row[1]["instance_id"], row[1]["x"], row[1]["y"]]))

    new_df = pd.DataFrame(
        [list(rows[i]) for i in range(len(rows))], columns=["instance_id", "x", "y"]
    )
    categorized_names = new_df["instance_id"].unique()
    gruped_points = {}

    for i in range(new_df["instance_id"].nunique()):
        selected_points = new_df.loc[new_df["instance_id"] == categorized_names[i]]
        gruped_points[categorized_names[i]] = {
            "x": selected_points["x"].to_numpy(),
            "y": selected_points["y"].to_numpy(),
        }

    fig, ax = plt.subplots()
    # ax.scatter(x_points, y_points, s=20)
    # df.plot.scatter(x= "x", y= "y", c= "instance_id", colormap="hsv")

    # df_scatter[['type', 'instance_id']] = df_scatter.instance_id.str.split(pat='_', expand=True)
    df_scatter["type_no"] = (
        pr.LabelEncoder()
        .fit(df_scatter["instance_id"].unique())
        .transform(df_scatter["instance_id"])
    )

    ax.scatter(df_scatter.x, df_scatter.y, c=df_scatter.type_no, cmap="hsv")
    ax.set_title("Simple plot")

    visualizate_points(categorized_names, gruped_points, ax, with_iterpolation=False)

    plt.show()


def get_normal_show(df):
    df["instance_id"] = df["instance_id"].apply(lambda x: x.split(sep="_")[0])

    x_points = df["x"].to_numpy()
    y_points = df["y"].to_numpy()

    categorized_names = df["instance_id"].unique()
    gruped_points = {}

    for i in range(df["instance_id"].nunique()):
        selected_points = df.loc[df["instance_id"] == categorized_names[i]]
        gruped_points[categorized_names[i]] = {
            "x": selected_points["x"].to_numpy(),
            "y": selected_points["y"].to_numpy(),
        }

    fig, ax = plt.subplots()
    ax.scatter(x_points, y_points, s=20)
    ax.set_title("Simple plot")

    visualizate_points(categorized_names, gruped_points, ax, with_iterpolation=True)

    plt.show()


def main(pierre=False):

    # prepare given points
    # df = pd.read_csv("./points/kk_swap_2d.csv", sep=';')
    df = pd.read_csv("./points/kamada_l1-mutual_attraction_2d.csv", sep=";")
    # df = pd.read_csv("./points/prefsynth_dataset/coordinates/mds_dapl2_2d.csv", sep=';')
    # df = pd.read_csv("./points/pierre.csv", sep=';')

    if pierre:
        get_pierre_syntetic(df)
    else:
        get_normal_show(df)


if __name__ == "__main__":
    pierre = False
    main(pierre)
