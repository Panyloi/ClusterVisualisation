import numpy as np
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import shapely.geometry as geometry


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


epsilon = 10e-12


def grahamAlgorithmUpgrade(points_pass):
    p0 = min(points_pass, key=lambda x: (x[1], x[0]))

    def orient(p, q, r):
        result = (p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0])
        if result > epsilon:
            return 1
        elif result < -epsilon:
            return -1
        else:
            return 0

    def partition(A, p, r):
        x = A[r]
        i = p - 1
        for j in range(p, r):
            result = orient(p0, A[j], x)
            if result == 1 or (result == 0 and distance(p0, A[j]) > distance(p0, x)):
                i += 1
                A[i], A[j] = A[j], A[i]

        A[i + 1], A[r] = A[r], A[i + 1]
        return i + 1

    def quicksort(A, p, r):
        if len(A) == 1:
            return A
        if p < r:
            q = partition(A, p, r)
            quicksort(A, p, q - 1)
            quicksort(A, q + 1, r)

    points = points_pass.copy()

    points.remove(p0)

    quicksort(points, 0, len(points) - 1)

    points = [p0] + points

    stack = []
    if len(points) < 3:
        return stack
    stack.append(points[0])
    stack.append(points[1])
    stack.append(points[2])

    i = 3
    while i < len(points):
        if orient(stack[-2], stack[-1], points[i]) == 1:

            stack.append(points[i])
            i = i + 1
        elif orient(stack[-2], stack[-1], points[i]) == 0:
            stack.pop()
            stack.append(points[i])
            i = i + 1
        else:
            stack.pop()

    if orient(stack[-2], stack[-1], p0) == 0:
        stack.pop()

    return stack


def concave_hull(points, alpha=0.0):
    print(points.shape)
    if len(points[0, :]) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return grahamAlgorithmUpgrade(points)

    def add_edge(edges, edge_points, coords, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array(list(zip(points[0, :], points[1, :])))
    # coords = points

    tri = Delaunay(coords)
    # print(f"Tri {tri.vertices}")
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for v1, v2, v3 in tri.vertices:
        pa = coords[v1]
        pb = coords[v2]
        pc = coords[v3]

        # Lengths of sides of triangle
        a = distance(pa, pb)
        b = distance(pb, pc)
        c = distance(pc, pa)

        s = (a + b + c) / 2.0
        area = np.sqrt(((s - a) * (s - b) * (s - c)) * s)

        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, v1, v2)
            add_edge(edges, edge_points, coords, v2, v3)
            add_edge(edges, edge_points, coords, v3, v1)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    print(unary_union)
    return unary_union(triangles), edge_points


if __name__ == "__main__":
    points = np.random.uniform(-20, -1, (2, 20))
    # print(points)
    points = np.array(
        [
            [
                -340.05424,
                -562.91549,
                -404.59063,
                -442.81896,
                -534.85711,
                -322.49817,
                -386.25334,
                -573.00877,
                -436.75252,
                -491.6311,
                -519.41224,
                -380.39107,
                -497.80467,
                -500.87839,
                -418.09673,
                -459.29183,
            ],
            [
                -506.45596,
                -363.55924,
                -500.48158,
                -378.46024,
                -421.95328,
                -541.66661,
                -414.21384,
                -265.36244,
                -425.48642,
                -404.36814,
                -312.07869,
                -510.35227,
                -446.7716,
                -358.2498,
                -465.63817,
                -455.99893,
            ],
        ]
    )
    print(f"max points: {points[0].max()} {points[1].max()}")
    print(f"min points: {points[0].min()} {points[1].min()}")
    points = points/100
    print(points)
    points_of_polygon, edge_points = concave_hull(points, alpha=0.3)
    fig, ax = plt.subplots()
    # ax.scatter(points[:, 0], points[:, 1])
    ax.scatter(points[0, :], points[1, :])

    xx, yy = points_of_polygon.exterior.coords.xy
    x, y = xx.tolist(), yy.tolist()
    line_of_polygon = list(zip(x, y))
    line_to_plot = [
        (
            line_of_polygon[j % len(line_of_polygon)],
            line_of_polygon[(j + 1) % len(line_of_polygon)],
        )
        for j in range(len(line_of_polygon))
    ]

    line_segments = LineCollection(segments=line_to_plot)
    ax.add_collection(line_segments)

    # print(f"Polygon points: {edge_points}")
    plt.show()
