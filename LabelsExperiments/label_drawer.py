import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from triesv2.kdt_c import calc

# ---------------------------------------------------------------------------- #
#                                GRAHAM ALGIRTHM                               #
# ---------------------------------------------------------------------------- #

epsilon = 10e-12
def grahamAlgorithmUpgrade(points_pass):
    p0 = min(points_pass, key=lambda x: (x[1],x[0]))

    def distance(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def orient(p,q,r):
        result = ((p[0] - r[0])*(q[1] - r[1]) - (p[1] - r[1])*(q[0] - r[0]))
        if result > epsilon:
            return 1
        elif result < -epsilon:
            return -1
        else:
            return 0

    def partition(A,p,r):
        x = A[r]
        i = p-1
        for j in range(p,r):
            result = orient(p0, A[j], x)
            if result == 1 or (result == 0 and distance(p0, A[j]) > distance(p0, x)):
                i+=1
                A[i],A[j] = A[j],A[i]

        A[i+1],A[r] = A[r],A[i+1]
        return i+1

    def quicksort(A,p,r):
        if len(A) == 1:
            return A
        if p < r:
            q = partition(A,p,r)
            quicksort(A,p,q-1)
            quicksort(A,q+1,r)

    points = points_pass.copy()

    points.remove(p0)

    quicksort(points, 0, len(points)-1)

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

# ---------------------------------------------------------------------------- #
#                                    DRAWER                                    #
# ---------------------------------------------------------------------------- #

def get_cmap(n, name='hsv'):
    cols = (plt.cm.get_cmap(name)
            (np.linspace(0, 1.0, n))[:, :3][:,::-1])
    return cols

df = pd.read_csv("./points/kk_swap_2d.csv", sep=';')

df["instance_id"] = df["instance_id"].apply(lambda x: x.split(sep="_")[0])

x_points = df["x"].to_numpy()
y_points = df["y"].to_numpy()

categorized_names = df["instance_id"].unique()
gruped_points = {}
fig, ax = plt.subplots()

for i in range(df["instance_id"].nunique()):
    selected_points = df.loc[df["instance_id"] == categorized_names[i]]
    gruped_points[categorized_names[i]] = {"x": selected_points["x"].to_numpy(),
                                           "y": selected_points["y"].to_numpy()}
    ax.scatter(gruped_points[categorized_names[i]]["x"], gruped_points[categorized_names[i]]["y"])

zipped_points = list(zip(x_points, y_points))
convex_hull_points = grahamAlgorithmUpgrade(zipped_points)
line_to_plot = [(convex_hull_points[j%len(convex_hull_points)], convex_hull_points[(j+1)%len(convex_hull_points)]) 
                for j in range(len(convex_hull_points))]
line_segments = LineCollection(segments = line_to_plot)
ax.add_collection(line_segments)

res = calc(gruped_points, convex_hull_points)
for set_name in res.keys():
    sres = res[set_name]
    pts = sres['points']
    pts.append(pts[0])
    x, y = zip(*pts)
    plt.plot(x, y)

    plt.plot([sres['tpoint'][0], sres['rpoint'][0]], [sres['tpoint'][1], sres['rpoint'][1]], color='black')

plt.show()