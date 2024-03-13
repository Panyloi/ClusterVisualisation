import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing, shgo, differential_evolution, direct
import time
from collections import defaultdict
from typing import List

tuple_add = lambda x, y: (x[0] + y[0], x[1] + y[1])


class Label:

    def __init__(self, x, y, width, height, root_point) -> None:

        # params
        self.x = x
        self.y = y

        # hiperparams
        self.rp_x = root_point[0]
        self.rp_y = root_point[1]
        self.width = width
        self.height = height

        # refreshable
        self.points = [tuple_add((x, y), (-self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, -self.height/2)),
                       tuple_add((x, y), (-self.width/2, -self.height/2))]

    def update_params(self, x, y):
        xd = x - self.x
        yd = y - self.y
        self.x = x
        self.y = y
        self.points = [tuple_add((xd, yd), point) for point in self.points]

    def hard_update(self, x, y):
        self.x = x
        self.y = y
        self.points = [tuple_add((x, y), (-self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, -self.height/2)),
                       tuple_add((x, y), (-self.width/2, -self.height/2))]

    def get_err(self):
        return np.sqrt((self.rp_x-self.x)**2 + (self.rp_y-self.y)**2)
    
    @classmethod
    def ll_set(cls, ll, x):
        for i, label in enumerate(ll):
            px, py = x[i*2], x[i*2+1]
            label.hard_update(px, py)
        


class MainSet:
    
    def __init__(self, plist) -> None:
        self.points = plist

        # error if to little points
        assert(len(self.points) > 2)

        self.edges = [(self.points[i], self.points[i+1]) for i in range(len(self.points) - 1)]
        self.edges.append((self.points[-1], self.points[0]))

        self.ul_point = (min(self.points, key=lambda x: x[0])[0], max(self.points, key=lambda x: x[1])[1])
        self.dr_point = (max(self.points, key=lambda x: x[0])[0], min(self.points, key=lambda x: x[1])[1])

    # intersection detecting method link https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    def ccw(self, Ax, Ay, Bx, By, Cx, Cy):
        return (Cy-Ay) * (Bx-Ax) > (By-Ay) * (Cx-Ax)

    def segment_intersect(self, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
        return self.ccw(Ax, Ay, Cx, Cy, Dx, Dy) != self.ccw(Bx, By, Cx, Cy, Dx, Dy) and \
               self.ccw(Ax, Ay, Bx, By, Cx, Cy) != self.ccw(Ax, Ay, Bx, By, Dx, Dy)

    def __contains__(self, point):
        x, y = point

        # easy case
        if x < self.ul_point[0]:
            return False
        elif x > self.dr_point[0]:
            return False
        elif y < self.dr_point[1]:
            return False
        elif y > self.ul_point[1]:
            return False

        intersections_count = 0

        for edge in self.edges:
            if (edge[0][1] > y and edge[1][1] > y) or (edge[0][1] < y and edge[1][1] < y):
                continue
            if self.segment_intersect(x, y, self.dr_point[0]+1, y, edge[0][0], edge[0][1], edge[1][0], edge[1][1]):
                intersections_count += 1

        return intersections_count % 2


class Grid:

    def __init__(self, div, plist):
        self.points = plist
        self.grid = defaultdict(lambda: defaultdict(list)) # main grid for storing points

        # division is made based on the main set range
        # x and y have different grid len
        # grid middle is in point (0, 0)
        self.ul_point = (min(self.points, key=lambda x: x[0])[0], max(self.points, key=lambda x: x[1])[1])
        self.dr_point = (max(self.points, key=lambda x: x[0])[0], min(self.points, key=lambda x: x[1])[1])

        self.x_glen = (self.dr_point[0] - self.ul_point[0])/div
        self.y_glen = (self.ul_point[1] - self.dr_point[1])/div

    def add(self, point):
        xi, yi = self.hash(point)
        self.grid[xi][yi].append(point)

    def get(self, ul_point, dr_point):
        
        # get raw grid indexes
        ul_xi, ul_yi = self.hash(ul_point)
        dr_xi, dr_yi = self.hash(dr_point)

        ret_points = []

        for xi in range(ul_xi, dr_xi + 1):
            for yi in range(ul_yi, dr_yi + 1):
                ret_points.extend(self.grid[xi][yi])

    def get_gen(self, ul_point, dr_point):
    
        # get raw grid indexes
        ul_xi, ul_yi = self.hash(ul_point)
        dr_xi, dr_yi = self.hash(dr_point)

        for xi in range(ul_xi, dr_xi + 1):
            for yi in range(ul_yi, dr_yi + 1):
                for p in self.grid[xi][yi]:
                    yield p

    def get_count(self, ul_point, dr_point):
        
        # get raw grid indexes
        ul_xi, ul_yi = self.hash(ul_point)
        dr_xi, dr_yi = self.hash(dr_point)

        c = 0
        for xi in range(ul_xi, dr_xi + 1):
            for yi in range(dr_yi, ul_yi + 1):
                c += len(self.grid[xi][yi])
        return c

    def clear(self):
        self.grid.clear()

    def hash(self, point):
        # hash function for points
        x, y = point
        xi = int(x // self.x_glen)
        yi = int(y // self.y_glen)

        return xi, yi



def loss(x, label_list: List[Label], main_set: MainSet, grid: Grid):
    # update structures
    grid.clear()

    for i, label in enumerate(label_list):
        label.update_params(x[i*2], x[i*2+1])
        for ps in label.points:
            grid.add(ps)

    # check for intersections between all labels
    intersections = 0
    loss = 0
    for label in label_list:
        intersections += grid.get_count(label.points[0], label.points[2]) - 4
        if any(point in main_set for point in label.points):
            intersections += 20 # consider hyperparametring the main set penalty (or adding multiline coeficient here)
        loss += label.get_err()
    
    return loss*(1 + (intersections)**3)


def present(ll: List[Label]):
    for label in ll:
        x, y = zip(*label.points)
        plt.plot(x, y)
    x, y = zip(*m.points)
    plt.plot(x, y)
    plt.show()
        
lnum = 10
mset = [(-2, -2), (-3, 0), (-2, 2), (0, 3), (5, 5), (3, 0), (2, -2), (0, -3)]
m = MainSet(mset)
g = Grid(5, mset)
ll = [Label(0, 0, 3, 1, (0, 0)) for _ in range(lnum)]
res = {}


def test_fun(fun: callable, pref="T"):
    start = time.time()
    res[pref] = fun(loss, bounds=[(-15, 15)]*lnum*2, args=(ll, m, g))
    end = time.time()
    print(end-start)

    # update result
    x = res[pref].x
    # Label.ll_set(ll, x)
    print(res[pref])
    print(x)
    present(ll)

test_fun(dual_annealing, pref="DA")
# test_fun(differential_evolution, pref="DE")
