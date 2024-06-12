from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.text import Text
import numpy as np
from scipy.optimize import dual_annealing, basinhopping
from scipy.spatial import KDTree, ConvexHull

from ..configuration import Configuration

CPoint = Tuple[float]
Point  = List[float]
CEdge  = Tuple[Tuple[float]]
Edge   = List[List[float]]

tuple_add = lambda x, y: (x[0] + y[0], x[1] + y[1])

# ---------------------------------------------------------------------------- #
#                                     LABEL                                    #
# ---------------------------------------------------------------------------- #

class Label:

    def __init__(self, text, r, a, width, height, root_point) -> None:

        # data
        self.text = text

        # params
        self.r = r
        self.a = a
        self.t = 0

        # hiperparams
        self.rp_x = root_point[0]
        self.rp_y = root_point[1]
        self.width = width
        self.height = height
        self.w2 = self.width/2
        self.h2 = self.height/2
        self.x0_x = None
        self.x0_y = None

        self.points = self.get_points()
        self.t_point = self.get_tpoint()
    
    def update_params(self, r, a, t):
        self.r = r
        self.a = a
        self.t = t
        self.points = self.get_points()
        self.t_point = self.get_tpoint()

    def get_points(self):
        x, y = self.get_point()
        return [tuple_add((x, y), (-self.w2, self.h2)),
                tuple_add((x, y), (self.w2, self.h2)),
                tuple_add((x, y), (self.w2, -self.h2)),
                tuple_add((x, y), (-self.w2, -self.h2))]

    def get_tpoint(self):
        if self.t // 1 == 0:
            return tuple_add(self.points[0], (self.width*(self.t % 1), 0))
        elif self.t // 1 == 1:
            return tuple_add(self.points[1], (0, -self.height*(self.t % 1)))
        elif self.t // 1 == 2:
            return tuple_add(self.points[2], (-self.width*(self.t % 1), 0))
        else:
            return tuple_add(self.points[3], (0, self.height*(self.t % 1)))
        
    def get_point(self):
        return self.rp_x + self.r*np.sin(self.a), self.rp_y + self.r*np.cos(self.a)
    
    def get_err(self, include_x0: bool = True):
        if include_x0:
            return (np.sqrt((self.rp_x-self.t_point[0])**2 + (self.rp_y-self.t_point[1])**2))
                    #  + \
                    # np.sqrt((self.x0_x - self.x)**2 + (self.x0_y - self.y)**2))/100
        else:
            return np.sqrt((self.rp_x-self.t_point[0])**2 + (self.rp_y-self.t_point[1])**2)
    
    def get_mpoint(self):
        x, y = self.get_point()
        return x, (y - self.h2) + self.w2
    
    def get_root_point(self) -> CPoint:
        return self.rp_x, self.rp_y
    
    def get_result(self):
        result = {'points': self.points,
                  'tpoint': self.t_point,
                  'rpoint': self.get_root_point(),
                  'point': self.get_point(),
                  'width' : self.width,
                  'height': self.height}
        return result
    
    def set_x0(self, x0_x, x0_y):
        self.x0_x, self.x0_y = x0_x, x0_y
    
    @classmethod
    def ll_get(cls, ll):
        res = []
        for label in ll:
            res.append(label.r)
            res.append(label.a)
        for label in ll:
            res.append(label.t)
        return np.array(res)
    
    @classmethod
    def ll_set(cls, ll, x: np.ndarray)-> None:
        for i, label in enumerate(ll):
            label.update_params(x[i*2], x[i*2+1], x[len(ll)*2+i])

    @classmethod
    def ll_set_x0(cls, ll, x0: np.ndarray) -> None:
        for i, label in enumerate(ll):
            label.set_x0(x0[i*2], x0[i*2+1])
        
# ---------------------------------------------------------------------------- #
#                                   MAIN SET                                   #
# ---------------------------------------------------------------------------- #

class MainSet:
    
    def __init__(self, plist: np.ndarray) -> None:
        self.points = plist

        # error if to little points
        assert(len(self.points) > 2)

        self.edges = [(self.points[i], self.points[i+1]) for i in range(len(self.points) - 1)]
        self.edges.append((self.points[-1], self.points[0]))

        maxx, maxy = plist.max(axis=0)
        minx, miny = plist.min(axis=0)

        self.ul_point = (float(minx), float(maxy))
        self.dr_point = (float(maxx), float(miny))

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

def loss(x: np.ndarray, label_list: List[Label], main_set: MainSet, include_x0_in_err: bool = True):
    
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
        if len(candidates[candidates[:, 1] <= label.points[3][1] + label.height]) > 4:
            intersections += 5

        # main set penalty
        if any(point in main_set for point in label.points):
            intersections += 15

        # loss
        loss += label.get_err(include_x0_in_err)

    return loss*((1 + intersections)**2)

def iter_loss(x: np.ndarray, label: Label, static_label_list: List[Label], main_set: MainSet, flg_arr: List[bool], include_x0_in_err: bool = False):
    
    # ---------------------------------- UPDATE ---------------------------------- #

    # update kd tree pts
    npts=[]
    for slabel in static_label_list:
        npts.extend(slabel.points) #TODO optimize this (make copy of existing set of points)

    # update label
    label.update_params(x[0], x[1], x[2])
    npts.extend(label.points)
        
    # create new KDTree
    nx = np.array(npts)
    kd = KDTree(nx)

    # ----------------------------------- LOSS ----------------------------------- #

    intersections = 0
    loss = 0

    for slabel in static_label_list:
        # static labels intersection
        mp = slabel.get_mpoint()
        candidates = nx[kd.query_ball_point(x=mp, r=slabel.width/2+0.001, p=np.inf)]
        if len(candidates[candidates[:, 1] <= slabel.points[3][1] + slabel.height]) > 4:
            flg_arr[0] = True
            intersections += 5

        # main set penalty
        if any(point in main_set for point in slabel.points):
            flg_arr[0] = True
            intersections += 15

    # label intersection
    mp = label.get_mpoint()
    candidates = nx[kd.query_ball_point(x=mp, r=label.width/2+0.001, p=np.inf)]
    if len(candidates[candidates[:, 1] <= label.points[3][1] + label.height]) > 4:
        flg_arr[0] = True
        intersections += 5

    # main set penalty
    if any(point in main_set for point in label.points):
        flg_arr[0] = True
        intersections += 15

    # loss
    loss += label.get_err(include_x0_in_err)

    return loss*((1 + intersections)**2)

        
# ---------------------------------------------------------------------------- #
#                                 HULL SWELLER                                 #
# ---------------------------------------------------------------------------- #


def swell_hull(hull_pts: np.ndarray, shift_mult: float):
    middle_point = np.mean(hull_pts, axis=0)
    new_pts = np.zeros_like(hull_pts)
    for i in range(len(hull_pts)):
        v = hull_pts[i] - middle_point
        new_pts[i] = hull_pts[i] + (v/np.linalg.norm(v))*shift_mult
    return new_pts


# ---------------------------------------------------------------------------- #
#                                    CALLER                                    #
# ---------------------------------------------------------------------------- #


def calc(data: dict, 
         points: np.ndarray,
         config_id: str):

    # Generate main set convex
    hull = ConvexHull(points)
    swelled_pts = swell_hull(points[hull.vertices], 10)

    mset = MainSet(swelled_pts)
    ll   = []

    fig, ax = plt.subplots()
    fig.add_axes(ax)
    fig.subplots_adjust(bottom=0.2)

    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)
    
    ax.bbox._bbox.x0 = 0.01
    ax.bbox._bbox.y0 = 0.15
    ax.bbox._bbox.x1 = 0.99
    ax.bbox._bbox.y1 = 0.99

    xlim = (Configuration.instance['editor']['init_xlim_low'], Configuration.instance['editor']['init_xlim_high'])
    ylim = (Configuration.instance['editor']['init_ylim_low'], Configuration.instance['editor']['init_ylim_high'])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for label_name in data.keys():
        ref_point = (data[label_name]['x'][0], data[label_name]['y'][0])

        tx: Text = ax.text(0, 0, label_name, size=Configuration.instance['editor']['font_size'], transform=ax.transData)
        tx.set_bbox(dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.3))
        text_bbox = tx.get_window_extent()
        transformed_text_bbox = Bbox(ax.transData.inverted().transform(text_bbox))

        ll.append(Label(label_name, 0, 1, transformed_text_bbox.width+4, transformed_text_bbox.height+4, ref_point))

    labels_bounds = [(0, 80), (0, np.pi*2)]*len(ll)
    labels_bounds.extend([(0, 4)]*len(ll))

    plt.clf()
    plt.close(fig)
    
    x=None

    if config_id == 'iterative':

        curr_config = Configuration.instance['labels_generator']['configurations'][config_id]
        maxiter = curr_config['maxiter']
        visit = curr_config['visit']
        initial_temp = curr_config['initial_temp']
        restart_temp_ratio = curr_config['restart_temp_ratio']
        accept = curr_config['accept']
        no_local_search = curr_config['no_local_search']

        res_ll = []
        for label in ll:
            label_bounds = [(0, 150), (0, np.pi*2), (0, 4)]
            flg = [False]
            lres = dual_annealing(iter_loss, bounds=label_bounds, args=(label, res_ll, mset, flg),
                                  maxiter=maxiter, visit=visit, initial_temp=initial_temp, restart_temp_ratio=restart_temp_ratio, 
                                  accept=accept, no_local_search=no_local_search)
            lx = lres.x

            flg[0] = False
            loss = iter_loss(lx, label, res_ll, mset, flg) # updates the label with finall x (and calculates finall intersection flag)
            res_ll.append(label)

        x = Label.ll_get(res_ll)
            

    elif config_id == 'global':

        x0 = None
        curr_config = Configuration.instance['labels_generator']['configurations'][config_id]
        if curr_config['generate_greedy_x0']:
            x0 = greedy((0, 0), ll, mset)
            Label.ll_set_x0(ll, x0)

        maxiter = curr_config['maxiter']
        visit = curr_config['visit']
        initial_temp = curr_config['initial_temp']
        restart_temp_ratio = curr_config['restart_temp_ratio']
        accept = curr_config['accept']
        no_local_search = curr_config['no_local_search']

        res = dual_annealing(loss, bounds=labels_bounds, args=(ll, mset), 
                             x0=x0, maxiter=maxiter, visit=visit, initial_temp=initial_temp, 
                             restart_temp_ratio=restart_temp_ratio, accept=accept, no_local_search=no_local_search)
        x = res.x
    elif config_id == 'divide_and_conquare':
        ...
    elif config_id == 'timed':
        ...
    else:
        raise Exception("Invalid config_id. No such configuration exists")

    # prep result
    assert(x is not None)
    labels = {}
    Label.ll_set(ll, x)
    for label in ll:
        labels[label.text] = label.get_result()

    return labels


def parse_solution_to_editor(labels: dict, state: dict) -> dict:
    
    for i, label_name in enumerate(labels.keys()):
        state['labels_data'][i] = {
            'text': label_name,
            'x': labels[label_name]['point'][0],
            'y': labels[label_name]['point'][1],
            'arrows': {
                0: {
                    'ref_x': labels[label_name]['rpoint'][0],
                    'ref_y': labels[label_name]['rpoint'][1],
                    'att_x': labels[label_name]['tpoint'][0],
                    'att_y': labels[label_name]['tpoint'][1],
                    'val': "a"
                }
            }
        }

    return state



def _debug_draw(ll: List[Label], points: np.ndarray):

    # Generate main set convex
    hull = ConvexHull(points)
    swelled_pts = swell_hull(points[hull.vertices], 2)
    mset = MainSet(swelled_pts)

    fig, ax = plt.subplots()
    fig.add_axes(ax)
    fig.subplots_adjust(bottom=0.2)

    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)
    
    ax.bbox._bbox.x0 = 0.01
    ax.bbox._bbox.y0 = 0.15
    ax.bbox._bbox.x1 = 0.99
    ax.bbox._bbox.y1 = 0.99

    xlim = (Configuration.instance['editor']['init_xlim_low'], Configuration.instance['editor']['init_xlim_high'])
    ylim = (Configuration.instance['editor']['init_ylim_low'], Configuration.instance['editor']['init_ylim_high'])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.plot(mset.points[:,0], mset.points[:,1], c='black')
    for label in ll:
        pt = label.get_point()
        tp = label.get_tpoint()
        rp  = label.get_root_point()
        txt = plt.text(pt[0], pt[1], label.text, size=Configuration.instance['editor']['font_size'], ha='center', va='center')
        txt.set_bbox(dict(boxstyle='round', pad=0.2, facecolor='white', edgecolor='black', alpha=0.2))
        raw_points = label.points
        raw_points.append(raw_points[0])
        raw_points = np.array(raw_points)
        plt.plot(raw_points[:,0], raw_points[:,1])
        plt.plot([tp[0], rp[0]], [tp[1], rp[1]])
    plt.show()
    

def _test():

    points = np.array([(-10, -10), (-10, 10), (10, 10), (10, -10)])

    ll   = [
        Label("tx1", 10*np.sqrt(2), np.pi/2 + np.pi/4, 4, 1, (0, 0)),
        Label("tx2", 13, np.pi/2 + np.pi/4, 4, 1, (0, 0))
    ]

    _debug_draw(ll, points)


#TODO:
# [x] try polar coordinates
# [x] iterative approach
# [x] hull swelling
# [ ] iterative with forced no intersection solution
# [ ] divide and conquare solution (by initial x0)
# [ ] iterative x0
# [ ] divide and conquare solution (after initial iterative solution, using final lines intersections)
