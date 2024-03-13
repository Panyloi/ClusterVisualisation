import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString
import psopy as pp


tuple_add = lambda x, y: (x[0] + y[0], x[1] + y[1])

class Label:

    def __init__(self, root_point, middle, width, height) -> None:

        # params
        self.pivot = middle
        self.line_param = 0

        # hiperparams
        self.root_point = root_point
        self.width = width
        self.height = height

        # dynamic
        self.line_point = None
        self.points = [tuple_add(self.pivot, (-self.width/2, self.height/2)),
                       tuple_add(self.pivot, (self.width/2, self.height/2)),
                       tuple_add(self.pivot, (self.width/2, -self.height/2)),
                       tuple_add(self.pivot, (-self.width/2, -self.height/2))]
        self._update_parametrized_point()   # must be after points update
        self.poly = Polygon(self.points)
        self.line = LineString([self.root_point, self.line_point])

    def _update_parametrized_point(self):
        if self.line_param // 1 == 0:
            self.line_point = tuple_add(self.points[0], (self.width*(self.line_param % 1), 0))
        elif self.line_param // 1 == 1:
            self.line_point = tuple_add(self.points[2], (0, self.height*(self.line_param % 1)))
        elif self.line_param // 1 == 2:
            self.line_point = tuple_add(self.points[3], (self.width*(1-(self.line_param % 1)), 0))
        else:
            self.line_point = tuple_add(self.points[3], (0, self.height*(1-(self.line_param % 1))))

    def set_params(self, pivot, line_param):
        self.pivot = pivot
        self.line_param = line_param

        # update
        self.points = [tuple_add(self.pivot, (-self.width/2, self.height/2)),
                       tuple_add(self.pivot, (self.width/2, self.height/2)),
                       tuple_add(self.pivot, (self.width/2, -self.height/2)),
                       tuple_add(self.pivot, (-self.width/2, -self.height/2))]
        self._update_parametrized_point() # must be after points update
        self.poly = Polygon(self.points)
        self.line = LineString([self.root_point, self.line_point])

    def get_err(self):
        # the after + is for the line start to prefer middle of the edges
        return self.line.length*(2 + abs((self.line_param % 1) - 0.5))
        

class MainSet:
    
    def __init__(self, *args) -> None:
        self.points = args
        self.poly = Polygon(self.points)


def loss(x, label_list, main_set):

    errors = 0

    # update lines
    for i in range(len(label_list)):
        label_list[i].set_params((x[i*3], x[i*3+1]), x[i*3+2])

    # check for intersections and update line cuts
    line_cuts = [0 for _ in range(len(label_list))]
    for i in range(len(label_list)-1):

        # check for main set intersection
        if label_list[i].poly.intersects(main_set.poly):
            errors += 5
        
        # check for labels intersections
        for j in range(i+1, len(label_list)):
            if label_list[i].poly.intersects(label_list[j].poly):
                errors += 3
            
            if label_list[i].line.intersects(label_list[j].poly):
                line_cuts[i] += 1

            if label_list[j].line.intersects(label_list[i].poly):
                line_cuts[j] += 1
    
    # intersection edgecase
    if label_list[-1].poly.intersects(main_set.poly):
        errors += 5
    
    loss = sum(line_cuts)

    for i in range(len(label_list)):
        loss += label_list[i].get_err()
    
    return loss*(1 + (1 + errors)**3)


def present(ll):
    for i in range(len(ll)):
        plt.plot(*ll[i].poly.exterior.xy)
        plt.plot(*ll[i].line.xy)
    plt.plot(*m.poly.exterior.xy)
    plt.show()
        
lnum = 8
m = MainSet((-2, -2), (-3, 0), (-2, 2), (0, 3), (5, 5), (3, 0), (2, -2), (0, -3))
ll = [Label((0, 0), (5, 5), 5, 2) for _ in range(lnum)]

x0 = np.random.uniform(-10, 10, (30, lnum*3))
res = pp.minimize(loss, x0, args=(ll, m))
print(res.x)
present(ll)
