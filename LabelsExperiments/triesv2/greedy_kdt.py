import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing, basinhopping
import time
from collections import defaultdict
from typing import List, Tuple, Optional
from scipy.spatial import KDTree

CPoint = Tuple[float]
Point  = List[float]
CEdge  = Tuple[Tuple[float]]
Edge   = List[List[float]]

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
            return tuple_add(self.points[1], (0, -self.height*(self.t % 1)))
        elif self.t // 1 == 2:
            return tuple_add(self.points[2], (-self.width*(self.t % 1), 0))
        else:
            return tuple_add(self.points[3], (0, self.height*(self.t % 1)))

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
        return np.sqrt((self.rp_x-self.t_point[0])**2 + (self.rp_y-self.t_point[1])**2)
    
    def get_mpoint(self):
        return self.x, (self.y - self.height/2) + self.width/2
    
    def get_root_point(self) -> CPoint:
        return (self.rp_x, self.rp_y)
    
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

    @classmethod
    def ll_get(cls, ll):
        res = []
        for label in ll:
            res.append(label.x)
            res.append(label.y)
        for label in ll:
            res.append(label.t)
        return np.array(res)
        
# ---------------------------------------------------------------------------- #
#                                   MAIN SET                                   #
# ---------------------------------------------------------------------------- #

class MainSet:
    
    def __init__(self, plist: List[Point]) -> None:
        self.points = plist

        # error if to little points
        assert(len(self.points) > 2)

        self.edges = [(self.points[i], self.points[i+1]) for i in range(len(self.points) - 1)]
        self.edges.append((self.points[-1], self.points[0]))

        self.ul_point = (min(self.points, key=lambda x: x[0])[0], max(self.points, key=lambda x: x[1])[1])
        self.dr_point = (max(self.points, key=lambda x: x[0])[0], min(self.points, key=lambda x: x[1])[1])

    # intersection detecting method link https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    # ---------------------------- LINES INTERSECTIONS --------------------------- #
        
    def intersection(self, Ax: float, Ay: float, Bx: float, By: float) -> float:
        # returns t parameter from which the intersection point is calculated like
        # [Bx-Ax, By-Ay]*t + [Ax, Ay]

        intersection_points = []
        for line in self.edges:
            t = MainSet._line_intersect(Ax, Ay, Bx, By, line[0][0], line[0][1], line[1][0], line[1][1])
            if t is not None:
                intersection_points.append(t)
        
        if not len(intersection_points):
            raise ValueError("No intersection found")

        return min(intersection_points)
    
    @staticmethod
    def _line_intersect(Ax1: float, Ay1: float, Ax2: float, Ay2: float, Bx1: float, By1: float, Bx2: float, By2: float) -> Optional[float]:
        # A is the base line and B is the line segment getting intersected (this matters because the base line is semi-infinite)
        d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
        if d:
            tA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
            tB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
        else:
            return None
        if not(0 <= tA and 0 <= tB <= 1):
            return None
        # x = Ax1 + tA * (Ax2 - Ax1)
        # y = Ay1 + tA * (Ay2 - Ay1)
        return tA

    # ----------------------------- POINT CONTAINING ----------------------------- #

    @staticmethod
    def _ccw(Ax: float, Ay: float, Bx: float, By: float, Cx: float, Cy: float) -> bool:
        return (Cy-Ay) * (Bx-Ax) > (By-Ay) * (Cx-Ax)

    @staticmethod
    def segment_intersect(Ax: float, Ay: float, Bx: float, By: float, Cx: float, Cy: float, Dx: float, Dy: float) -> bool:
        return MainSet._ccw(Ax, Ay, Cx, Cy, Dx, Dy) != MainSet._ccw(Bx, By, Cx, Cy, Dx, Dy) and \
               MainSet._ccw(Ax, Ay, Bx, By, Cx, Cy) != MainSet._ccw(Ax, Ay, Bx, By, Dx, Dy)

    def __contains__(self, point: Point) -> bool:
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
#                                    GREEDY                                    #
# ---------------------------------------------------------------------------- #
    
def greedy(middle_point: Point, ll: List[Label], ms: MainSet) -> np.ndarray:
    mpx, mpy = middle_point

    for label in ll:
        # calculate label intersection parameter
        lpx, lpy = label.get_root_point()
        t = ms.intersection(mpx, mpy, lpx, lpy)
        nx, ny = mpx + t*(lpx-mpx), mpy + t*(lpy-mpy)

        # set new label parameters based on what positioning
        if lpy >= mpy and lpx >= mpx:
            # upper right
            label.update_params(nx + label.width/2, ny + label.height/2, 3)
        elif lpy >= mpy and lpx < mpx:
            # upper left
            label.update_params(nx - label.width/2, ny + label.height/2, 2)
        elif lpy < mpy and lpx >= mpx:
            # lower right
            label.update_params(nx + label.width/2, ny - label.height/2, 0)
        else:
            # lower left
            label.update_params(nx - label.width/2, ny - label.height/2, 1)

    return Label.ll_get(ll)

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
            intersections += 5

        # main set penalty
        if any(point in main_set for point in label.points):
            intersections += 15

        # loss
        loss += label.get_err()

    return loss*((1 + intersections)**2)

        
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
    x0 = greedy((0, 0), ll, mset)
    # res = basinhopping(loss, x0, minimizer_kwargs={"args" : (ll, mset)})
    res = dual_annealing(loss, bounds=labels_bounds, args=(ll, mset), x0=x0, maxiter=4000, initial_temp=1000, visit=1.65)
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
