import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing, shgo, differential_evolution, direct
import time
from collections import defaultdict
from typing import List
from scipy.spatial import KDTree

tuple_add = lambda x, y: (x[0] + y[0], x[1] + y[1])

# ---------------------------------------------------------------------------- #
#                                     LABEL                                    #
# ---------------------------------------------------------------------------- #

class Label:

    def __init__(self, text, x, y, width, height, root_point) -> None:

        # data
        self.text = text

        # params
        self.x = x
        self.y = y
        self.t = 0

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
        
        self.t_point = self.get_tpoint()
    
    def get_tpoint(self):
        if self.t // 1 == 0:
            return tuple_add(self.points[0], (self.width*(self.t % 1), 0))
        elif self.t // 1 == 1:
            return tuple_add(self.points[2], (0, self.height*(self.t % 1)))
        elif self.t // 1 == 2:
            return tuple_add(self.points[3], (self.width*(1-(self.t % 1)), 0))
        else:
            return tuple_add(self.points[3], (0, self.height*(1-(self.t % 1))))

    def update_params(self, x, y, t):
        self.x = x
        self.y = y
        self.points = [tuple_add((x, y), (-self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, self.height/2)),
                       tuple_add((x, y), (self.width/2, -self.height/2)),
                       tuple_add((x, y), (-self.width/2, -self.height/2))]
        self.t = t
        self.t_point = self.get_tpoint()

    def get_err(self):
        return np.sqrt((self.rp_x-self.t_point[0])**2 + (self.rp_y-self.t_point[1])**2 + (self.rp_x - self.x)**2 + (self.rp_y - self.y)**2)
    
    def get_mpoint(self):
        return self.x, (self.y - self.height/2) + self.width/2
    
    def get_result(self):
        result = {'points': self.points,
                  'tpoint': self.t_point,
                  'rpoint': (self.rp_x, self.rp_y)}
        return result
    
    @classmethod
    def ll_set(cls, ll, x):
        for i, label in enumerate(ll):
            px, py = x[i*2], x[i*2+1]
            label.update_params(px, py, x[len(ll)*2+i])
        
# ---------------------------------------------------------------------------- #
#                                   MAIN SET                                   #
# ---------------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------------- #
#                                     LOSS                                     #
# ---------------------------------------------------------------------------- #

def loss(x: np.ndarray, label_list: List[Label], main_set: MainSet):
    
    # ---------------------------------- UPDATE ---------------------------------- #

    # update label parameters
    npts=[]
    for i, label in enumerate(label_list):
        label.update_params(x[i*2], x[i*2+1], x[len(label_list)*2 + i])
        npts.extend(label.points)
        
    # create new KDTree
    nx = np.array(npts)
    kd = KDTree(nx)

    # ----------------------------------- LOSS ----------------------------------- #

    intersections = 0
    loss = 0
    for label in label_list:
        # labels intersection
        mp = label.get_mpoint()
        candidates = nx[kd.query_ball_point(x=mp, r=label.width/2+0.001, p=np.inf)]
        if len(candidates[candidates[:, 1] <= label.y + label.height/2]) > 4:
            intersections += 30

        # main set penalty
        if any(point in main_set for point in label.points):
            intersections += 80

        # loss
        loss += label.get_err()

    return loss*((1 + intersections)**3)

        
# ---------------------------------------------------------------------------- #
#                                    CALLER                                    #
# ---------------------------------------------------------------------------- #

def calc(grouped_points: dict, convex_points: list):
    mset     = MainSet(convex_points)
    ll       = []
    ul_point = (min(convex_points, key=lambda x: x[0])[0], max(convex_points, key=lambda x: x[1])[1])
    dr_point = (max(convex_points, key=lambda x: x[0])[0], min(convex_points, key=lambda x: x[1])[1])
    xw       = (dr_point[0] - ul_point[0])*0.1
    yw       = (ul_point[1] - dr_point[1])*0.1

    for label_name in grouped_points.keys():
        ref_point = (grouped_points[label_name]['x'][0], grouped_points[label_name]['y'][0])
        ll.append(Label(label_name, 0, 0, len(label_name)*yw*0.4/1.6, yw*0.4, ref_point))

    labels_bounds = [(ul_point[0]-xw, dr_point[0]+xw), (dr_point[1]-yw, ul_point[1]+yw)]*len(ll)
    labels_bounds.extend([(0, 4)]*len(ll))

    # calculate
    start = time.time()
    res = dual_annealing(loss, bounds=labels_bounds, args=(ll, mset), maxiter=1000, initial_temp=10000, visit=2.9)
    end = time.time()
    print(end-start)
    print(res)

    # prep result
    labels = {}
    Label.ll_set(ll, res.x)
    for label in ll:
        label: Label
        labels[label.text] = label.get_result()
    return labels
