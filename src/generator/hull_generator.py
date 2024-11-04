from copy import deepcopy
from collections import deque

import numpy as np
from scipy import interpolate
from scipy.spatial.distance import pdist

from concave_hull import concave_hull_indexes

epsilon = 10e-8


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

def calculate_angle(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    
    magnitude_BA = np.sqrt(BA[0]**2 + BA[1]**2)
    magnitude_BC = np.sqrt(BC[0]**2 + BC[1]**2)
    
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def handle_corners_points(points, default_points_dict, num_of_points=10):
    if len(points) <= 3:
        return points

    num_of_sets = len(points)
    after_curve = False
    visited_points = {}

    bourder_value = int(
        num_of_points * 0.5
    )  # 0.3 is hyperparapeter that euqals 30% of all points (TODO: adjust this value in further attmpts)

    new_points = []
    points_queue = deque()
    straight_points_list = [0, 0, 1]  # 1 == straight_points
    points_in_straight_line = []
    test_point = None

    p0 = min(points[0], key=lambda x: (x[1], x[0]))
    points_queue.append(default_points_dict[(p0[0], p0[1])])
    test_point = default_points_dict[
        (p0[0], p0[1])
    ]  # required value for cicuit of main point is grather than 6 !!! (TODO: check this)

    i_array = 0
    second_point = None
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

                # if points_union[i] in next_points_array:
                second_point = last_point
                last_point = points_union[i]


        # calculated_angle = calculate_angle(current_point, last_point, second_point)
        # print(calculated_angle)
        current_point = last_point
        if default_points_dict[(current_point[0], current_point[1])] not in visited_points.keys():
            visited_points[default_points_dict[(current_point[0], current_point[1])]] = 1
        # print(f"MOCNE COS {default_points_dict[(current_point[0], current_point[1])]}")
        # print(current_point)
        if i_array > num_of_sets + 2:
            break
        
        # if np.abs(calculated_angle) < 20:
        #     continue


        if current_point in current_points_array:
            straight_points_list[1] += 1
            points_in_straight_line.append(current_point)
        else:
            _point = default_points_dict[(current_point[0], current_point[1])]
            if len(points_queue) == 3:  # "if" is to handle first start of function
                test_point = points_queue.popleft()
                if (
                    _point == test_point and straight_points_list[1] > bourder_value
                ) or _point != test_point:
                    if straight_points_list[1] > bourder_value:
                        after_curve = True

                        new_points.append(current_points_array)
                    new_points.append(current_points_array)
                elif _point != test_point and visited_points[_point] == 0:
                    visited_points[_point] += 1
                    new_points.append(current_points_array)

                if after_curve:
                    # print(f"JAK TO TAK {default_points_dict[(current_point[0], current_point[1])]}")
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
    circuit_points = {}  # key: points from circle; value: point from orginal data
    main_points_of_circuit = (
        {}
    )  # key: point from orginal data; value: points from circle

    # circle
    angles = np.linspace(0, 2 * np.pi, num_of_points, endpoint=True)

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

    return (
        np.array(points),
        handle_corners_points(new_circuit, circuit_points, num_of_points),
        circuit_points,
        main_points_of_circuit,
    )


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
def greedy_selecting_1(points):
    if len(points) <= 3:
        points_transform = []
        for i in range(len(points)):
            points_transform.extend(points[i])
        idxes = concave_hull_indexes(
                points_transform,
                concavity=float("inf"),
                length_threshold=0,
            )
        
        _idx_points = [points_transform[idxes[j]] for j in range(len(idxes))]

        return _idx_points

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

def domain_expansion(
    points_to_transform: list[tuple[float, float]],
    circle_points,
    drawing_point_radius=1.5,
    multiplier=2,
):

    result_points = []

    for point in points_to_transform:
        main_point = circle_points[(point[0], point[1])]
        a_b_distance = distance(main_point, point)

        x_c_edge = main_point[0] + (drawing_point_radius * multiplier) * (
            (point[0] - main_point[0]) / a_b_distance
        )
        y_c_edge = main_point[1] + (drawing_point_radius * multiplier) * (
            (point[1] - main_point[1]) / a_b_distance
        )
        c_edge = (x_c_edge, y_c_edge)

        new_c = (2 * c_edge[0] - main_point[0], 2 * c_edge[1] - main_point[1])
        result_points.append(new_c)

    return result_points


def sort_elements_in_concave_hull(points):
    min_point_by_y = min(points, key=lambda point: (point[1], point[0]))
    i = 0

    for point in points:
        if point[0] == min_point_by_y[0] and point[1] == min_point_by_y[1]:
            break
        i += 1

    new_points = []
    for _ in range(len(points)):
        new_points.append(points[i % len(points)])
        i += 1

    return new_points


def the_same_circle(point1, point2):
    if point1[0] == point2[0]:
        if point1[1] == point2[1]:
            return True
    return False


def get_all_other_intersect_circles(points, point, dist):
    result = []

    for _point in points:
        if circle_intersect(_point, point, dist) and not the_same_circle(_point, point):
            result.append(_point)
    return result


def join_closest_points(points, circle_points_to_main, main_points_to_circle, dist, segment_length):
    

    new_points = []
    empty_sets = True
    not_same_points = True
    points_to_join = []
    points_transform = []
    visited_points = set()


    for i in range(len(points) - 1):
        first_point = circle_points_to_main[points[i][0]]
        second_point = circle_points_to_main[points[i+1][0]]
        
        if circle_intersect(first_point, second_point, dist):
            empty_sets = False
            points_to_join.append(first_point)
        else:
            if not empty_sets:
                points_to_join.append(first_point)
                for point_to_join in points_to_join:
                    not_same_points = False
                    
                    if point_to_join[0] != points_to_join[0][0] and point_to_join[1] != points_to_join[0][1]:
                        not_same_points = True  
                
                if not_same_points:
                    
                    for point_to_join in points_to_join:
                        points_transform.extend(main_points_to_circle[point_to_join])

                    idxes = concave_hull_indexes(
                            points_transform,
                            concavity=float("inf"),
                            length_threshold=0,
                        )
                    
                    _idx_points = [points_transform[idxes[j]] for j in range(len(idxes))]
                    new_points.append(_idx_points)
                else:
                    for point_to_join in points_to_join:
                        new_points.append(main_points_to_circle[point_to_join])

                points_transform = []
                points_to_join = []
                empty_sets = True
                not_same_points = True
            else:
                new_points.append(points[i])
                points_to_join = []

    new_points.append(points[-1])
    return new_points

def join_closest_points_copy(points, circle_points_to_main, main_points_to_circle, dist, segment_length):
    

    # empty_sets = True
    # not_same_points = True
    # points_to_join = []
    
    points_transform = []
    new_points = []
    visited_points = set()
    curr_points = set()
    hull_points = list(main_points_to_circle.keys())
    Q = deque()

    for i in range(len(points)):
        first_point = circle_points_to_main[points[i][0]]
        curr_points = set()
        points_transform = []
        Q.append(first_point)

        while Q:
            q_point = Q.popleft()
            
            if q_point in visited_points:
                continue

            curr_points.add(q_point)
            visited_points.add(q_point)
            
            curr_intersect = get_all_other_intersect_circles(hull_points, q_point, dist)
            
            for _point in curr_intersect:
                if _point not in visited_points:
                    Q.append(_point)

        if len(curr_points) == 0:
            continue
        elif len(curr_points) == 1:
            new_points.append(points[i])
        else:
            for curr_point in curr_points:
                points_transform.extend(main_points_to_circle[curr_point])

            idxes = concave_hull_indexes(
                    points_transform,
                    concavity=float("inf"),
                    length_threshold=0,
                )
            
            _idx_points = [points_transform[idxes[j]] for j in range(len(idxes))]
            new_points.append(_idx_points)
    
    return new_points
        
        
    



def calc_hull(
    data: dict, circle_radious: float, points_in_circle: int, segment_length: int
):

    hulls = {}

    for i, hull_name in enumerate(data.keys()):

        hulls[i] = {}

        hulls[i]["name"] = hull_name
        print(hull_name)

        x = deepcopy(data[hull_name]["x"])
        y = deepcopy(data[hull_name]["y"])
        # x = np.repeat(x, 2)
        # y = np.repeat(y, 2)
        hulls[i]["cluster_points"] = {"x": x, "y": y}

        points_transform = np.hstack(
            (
                x.reshape(-1, 1),
                y.reshape(-1, 1),
            )
        )

        # try:
        #     distances = pdist(np.unique(points_transform, axis=0))
        #     print(f"DISTANCES {np.min(distances)}")
        # except Exception:
        #     pass

        points_transform = np.unique(points_transform, axis=0)

        idxes = concave_hull_indexes(
            points_transform[:, :2],
            concavity=1,
            length_threshold=0,
        )

        _idx_points = [points_transform[idxes[i]] for i in range(len(idxes))]
        _idx_points = sort_elements_in_concave_hull(_idx_points)

        points_transform, to_jarvis, circuit_points, main_points_of_circuit = (
            convert_points(_idx_points, circle_radious, num_of_points=points_in_circle)
        )

        to_jarvis = join_closest_points(to_jarvis, circuit_points, main_points_of_circuit, 1, 0.1)

        selected_points = greedy_selecting_1(to_jarvis)

        _idx_points = domain_expansion(
            selected_points, circuit_points, drawing_point_radius=1.5, multiplier=1
        )
        _idx_points = make_dense_points(_idx_points, segment_length=segment_length)

        hulls[i]["polygon_points"] = _idx_points

        _, _idx_points_unique = np.unique(_idx_points, axis=0, return_index=True)

        _idx_points = np.array(_idx_points)
        _idx_points = _idx_points[np.sort(_idx_points_unique)]
        _idx_points = interpolate_points(_idx_points, 1000)
        
        hulls[i]["interpolate_points"] = _idx_points

        # _idx_points = interpolate_points(_idx_points[1:], 1000)

        polygon_lines = [
            (
                _idx_points[j % len(_idx_points)],
                _idx_points[(j + 1) % len(_idx_points)],
            )
            for j in range(len(_idx_points))
        ]
        print(len(polygon_lines))
        hulls[i]["polygon_lines"] = polygon_lines

    return hulls


def set_hull_parameters(
    state, circle_radious: float, points_in_circle: int, segment_length: int
):
    state["hulls_data"]["parameters"] = {
        "circle_radious": circle_radious,
        "points_in_circle": points_in_circle,
        "segment_length": segment_length,
    }


def calc_one_hull(hull_name, points, state):
    hulls = {}

    hulls[hull_name] = {}

    x = np.array(points["x"])
    y = np.array(points["y"])
    # x = np.repeat(x, 2)
    # y = np.repeat(y, 2)
    hulls[hull_name]["cluster_points"] = {"x": x, "y": y}
    print(f"HULL NAME {hull_name}")
    points_transform = np.hstack(
        (
            x.reshape(-1, 1),
            y.reshape(-1, 1),
        )
    )
    points_transform = np.unique(points_transform, axis=0)
    

    idxes = concave_hull_indexes(
        points_transform[:, :2],
        concavity=1,
        length_threshold=0,
    )

    _idx_points = [points_transform[idxes[i]] for i in range(len(idxes))]
    _idx_points = sort_elements_in_concave_hull(_idx_points)

    points_transform, to_jarvis, circuit_points, main_points_of_circuit = convert_points(
        _idx_points,
        state["hulls_data"]["parameters"]["circle_radious"],
        num_of_points=state["hulls_data"]["parameters"]["points_in_circle"],
    )

    to_jarvis = join_closest_points_copy(to_jarvis, circuit_points, main_points_of_circuit, 1.5, 0.1)


    # to_jarvis = handle_corners_points(to_jarvis, circuit_points,state["hulls_data"]["parameters"]["points_in_circle"])
    if hull_name == "nauty-72":  
        print(to_jarvis)

    selected_points = greedy_selecting_1(to_jarvis)

    _idx_points = domain_expansion(
            selected_points, circuit_points, drawing_point_radius=1.5, multiplier=1
        )

    _idx_points = make_dense_points(
        _idx_points,
        segment_length=state["hulls_data"]["parameters"]["segment_length"],
    )

    hulls[hull_name]["polygon_points"] = _idx_points

    _, _idx_points_unique = np.unique(_idx_points, axis=0, return_index=True)

    _idx_points = np.array(_idx_points)
    _idx_points = _idx_points[np.sort(_idx_points_unique)]
    _idx_points = interpolate_points(_idx_points, 1000)
    
    hulls[hull_name]["interpolate_points"] = _idx_points
    
    polygon_lines = [
        (
            _idx_points[j % len(_idx_points)],
            _idx_points[(j + 1) % len(_idx_points)],
        )
        for j in range(len(_idx_points))
    ]

    hulls[hull_name]["polygon_lines"] = polygon_lines

    return hulls


def parse_solution_to_editor_hull(hulls: dict, state: dict) -> dict:
    state["hulls_data"]["hulls"] = {}
    for i in hulls.keys():
        # state['hulls_data'][i] = {
        #     'name': hulls[i]['name'],
        #     'cords': hulls[i]['polygon_points'],
        #     'line_cords': hulls[i]['polygon_lines'],
        #     'cluster_points': hulls[i]['cluster_points']
        # }
        state["hulls_data"]["hulls"][hulls[i]["name"]] = {
            "cords": hulls[i]["polygon_points"],
            "line_cords": hulls[i]["polygon_lines"],
            "cluster_points": hulls[i]["cluster_points"],
            "interpolate_points": hulls[i]["interpolate_points"]
        }
        state["hulls_data"]["change"] = {}
        state["hulls_data"]["undraw"] = set()
        state["hulls_data"][hulls[i]["name"]] = dict()
        state["hulls_data"][hulls[i]["name"]]["hull_line"] = dict()
        
        for j, line in enumerate(hulls[i]["polygon_lines"]):
            state["hulls_data"][hulls[i]["name"]]["hull_line"][j] = {
                'x1': line[0][0],
                'y1': line[0][1],
                'x2': line[1][0],
                'y2': line[1][1],
                'val': hulls[i]["name"]
            }


    return state
