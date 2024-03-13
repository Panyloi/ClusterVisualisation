import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing, shgo, differential_evolution, direct
import time
from collections import defaultdict
from typing import List
from scipy.spatial import KDTree

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
        self.x = x
        self.y = y
        self.points = [tuple_add((x, y), (-self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, -self.height/2)),
                       tuple_add((x, y), (-self.width/2, -self.height/2))]

    def get_err(self):
        return np.sqrt((self.rp_x-self.x)**2 + (self.rp_y-self.y)**2)
    
    def get_mpoint(self):
        return self.x, (self.y - self.height/2) + self.width/2
    
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



def loss(x: np.ndarray, label_list: List[Label], main_set: MainSet):
    
    # ---------------------------------- UPDATE ---------------------------------- #

    # update label parameters
    for i, label in enumerate(label_list):
        label.update_params(x[i*2], x[i*2+1])
        
    # create new KDTree
    nx = x.reshape(-1, 2)
    kd = KDTree(nx)

    # ----------------------------------- LOSS ----------------------------------- #

    intersections = 0
    loss = 0
    for label in label_list:
        # labels intersection
        mp = label.get_mpoint()
        candidates = nx[kd.query_ball_point(x=mp, r=label.width/2, p=np.inf, workers=-1)]
        intersections += 2*len(candidates[candidates[:, 1] < label.y + label.height/2])

        # main set penalty
        if any(point in main_set for point in label.points):
            intersections += 20

        # loss
        loss += label.get_err()
    
    return loss*(1 + (intersections)**3)


def present(ll: List[Label]):
    for label in ll:
        x, y = zip(*label.points)
        plt.plot(x, y)
    x, y = zip(*m.points)
    plt.plot(x, y)
    plt.show()
        
lnum = 16
mset = [(-2, -2), (-3, 0), (-2, 2), (0, 3), (5, 5), (3, 0), (2, -2), (0, -3)]
m = MainSet(mset)
ll = [Label(0, 0, 3, 1, (0, 0)) for _ in range(lnum)]
res = {}


def test_fun(fun: callable, pref="T"):
    start = time.time()
    res[pref] = fun(loss, bounds=[(-15, 15)]*lnum*2, args=(ll, m))
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
