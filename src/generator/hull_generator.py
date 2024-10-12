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


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def fill_line_with_points(points, pointA, pointB, segment_length=8):
    point1 = np.array(pointA)
    point2 = np.array(pointB)

    distance = np.linalg.norm(point2 - point1)

    num_segments = int(np.ceil(distance / segment_length))

    for i in range(num_segments):
        new_point = list(point1 + i * (point2 - point1) / num_segments)
        points.append(tuple(new_point))


def make_dense_points(points, segment_length=10):
    new_points = []
    for i in range(len(points) - 1):
        if distance(points[i], points[i + 1]) > segment_length:
            fill_line_with_points(new_points, points[i], points[i + 1], segment_length)
        else:
            new_points.append(points[i])
    new_points.append(tuple(points[-1]))
    return new_points


def interpolate_points(points, num=100):
    if len(points) < 3:
        return points
    u3 = np.linspace(0, 1, num, endpoint=True)

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


def circle_intersect(O1, O2, R):
    d = np.sqrt((O1[0] - O2[0]) * (O1[0] - O2[0]) + (O1[1] - O2[1]) * (O1[1] - O2[1]))
    return d <= (2 * R)


def handle_corners_points(points, default_points_dict, num_of_points=10):
    if len(points) <= 1:
        return points

    num_of_sets = len(points)
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

        if i_array > num_of_sets + 2:
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
        concavity=2,  # TODO: check other numbers for parameter
        length_threshold=0,
    )

    x, y = points[idxes[0]]
    current_point = dict_circuit_points[(x, y)]

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

        jarvis_points.extend(new_set_of_points)

    new_circuit = create_new_circuit(
        jarvis_points, circuit_points, main_points_of_circuit
    )

    return np.array(points), handle_corners_points(new_circuit, circuit_points)


def orient(p, q, r):
    result = (p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0])
    if result > epsilon:
        return 1
    elif result < -epsilon:
        return -1
    else:
        return 0


"""
params:
    points : set of circle points 
             [[(x11, y11), (x21, y21), ..., (xn1, yn1)], [(x12, y12), (x22, y22), ..., (xm2, ym2)], ...]

return:
    points that create outline for cluster points

"""


def greedy_selecting(points):
    if len(points) <= 1:
        return points[0]

    num_of_sets = len(points)
    first_points_set = set(points[0])
    second_points_set = set(points[1])

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
                    second_point = points_union[i]

                if current_points_array == second_next_points_array:
                    last_point = second_point
                else:
                    last_point = points_union[i]

        current_point = last_point

        # if second_point in first_points_set:
        if i_array >= num_of_sets + 2:
            break

        if current_point in current_points_array:
            straight_points += 1

        if current_point in next_points_array:
            straight_points = 0

            i_array += 1

            current_points_array = next_points_array
            next_points_array = second_next_points_array
            second_next_points_array = points[i_array % len(points)]

        elif current_point in second_next_points_array:
            straight_points = 0

            i_array += 1
            current_points_array = second_next_points_array
            next_points_array = points[i_array % len(points)]

            i_array += 1
            second_next_points_array = points[i_array % len(points)]

        outline_points.append(current_point)

    return outline_points


def calc_hull(data: dict, circle_radious: float, points_in_circle: int, segment_length: int):

    hulls = {}

    for i, hull_name in enumerate(data.keys()):

        

        hulls[i] = {}
        
        hulls[i]['name'] = hull_name

        x = deepcopy(data[hull_name]['x'])
        y = deepcopy(data[hull_name]['y'])

        hulls[i]['cluster_points'] = {'x': x, 'y': y}

        

        points_transform = np.hstack(
            (
                x.reshape(-1, 1),
                y.reshape(-1, 1),
            )
        )

        

        idxes = concave_hull_indexes(
            points_transform[:, :2],
            concavity=2,
            length_threshold=0,
        )

        _idx_points = [points_transform[idxes[i]] for i in range(len(idxes))]
        
        if hull_name=="Interval":
            print(_idx_points)

        points_transform, to_jarvis = convert_points(
            _idx_points, circle_radious, num_of_points=points_in_circle
        )

        _idx_points = make_dense_points(
            greedy_selecting(to_jarvis),
            segment_length=segment_length
        )

        hulls[i]['polygon_points'] = _idx_points 

        _idx_points = interpolate_points(_idx_points, 1000)

        polygon_lines = [
            (
                _idx_points[j % len(_idx_points)],
                _idx_points[(j + 1) % len(_idx_points)],
            )
            for j in range(len(_idx_points))
        ]

        hulls[i]['polygon_lines'] = polygon_lines

    return hulls

def set_hull_parameters(state, circle_radious: float, points_in_circle: int, segment_length: int):
    state['hulls_data']['parameters'] = {
        'circle_radious': circle_radious,
        'points_in_circle': points_in_circle,
        'segment_length': segment_length,
    }

def calc_one_hull(hull_name, points, state):
    hulls = {}


    hulls[hull_name] = {}

    x = np.array(points['x'])
    y = np.array(points['y'])

    hulls[hull_name]['cluster_points'] = {'x': x, 'y': y}

    points_transform = np.hstack(
        (
            x.reshape(-1, 1),
            y.reshape(-1, 1),
        )
    )

    idxes = concave_hull_indexes(
        points_transform[:, :2],
        concavity=3,
        length_threshold=0,
    )

    _idx_points = [points_transform[idxes[i]] for i in range(len(idxes))]

    points_transform, to_jarvis = convert_points(
        _idx_points, 
        state['hulls_data']['parameters']['circle_radious'],
        num_of_points=state['hulls_data']['parameters']['points_in_circle']
    )

    _idx_points = make_dense_points(
        greedy_selecting(to_jarvis),
        segment_length=state['hulls_data']['parameters']['segment_length']
    )

    hulls[hull_name]['polygon_points'] = _idx_points 

    _idx_points = interpolate_points(_idx_points, 100)

    polygon_lines = [
        (
            _idx_points[j % len(_idx_points)],
            _idx_points[(j + 1) % len(_idx_points)],
        )
        for j in range(len(_idx_points))
    ]

    hulls[hull_name]['polygon_lines'] = polygon_lines

    return hulls

def parse_solution_to_editor_hull(hulls: dict, state: dict) -> dict:
    state['hulls_data']['hulls'] = {}
    for i in hulls.keys():
        # state['hulls_data'][i] = {
        #     'name': hulls[i]['name'],
        #     'cords': hulls[i]['polygon_points'],
        #     'line_cords': hulls[i]['polygon_lines'],
        #     'cluster_points': hulls[i]['cluster_points']
        # }
        state['hulls_data']['hulls'][hulls[i]['name']] = {
            'cords': hulls[i]['polygon_points'],
            'line_cords': hulls[i]['polygon_lines'],
            'cluster_points': hulls[i]['cluster_points']
        }
        state['hulls_data']['change'] = {}
        state['hulls_data']['undraw'] = set()
        
    return state


