import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from algorithms.convex_hull import concave_hull

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

for i in range(df["instance_id"].nunique()):
    selected_points = df.loc[df["instance_id"] == categorized_names[i]]
    gruped_points[categorized_names[i]] = {"x": selected_points["x"].to_numpy(),
                                           "y": selected_points["y"].to_numpy()}
fig, ax = plt.subplots()

# ax.scatter(gruped_points[categorized_names[i]]["x"], gruped_points[categorized_names[i]]["y"])
ax.scatter(x_points, y_points, s=1)
ax.set_title('Simple plot')

cmap = get_cmap(len(categorized_names))
np.random.shuffle(cmap)

for i in range(len(categorized_names)):
    zipped_points = list(zip(gruped_points[categorized_names[i]]["x"], gruped_points[categorized_names[i]]["y"]))

    # convex_hull_points = grahamAlgorithmUpgrade(zipped_points)
    convex_hull_points, edge_points = concave_hull(zipped_points, alpha=0.6)

    if len(convex_hull_points) < 3:
        continue

    # line_to_plot = [(convex_hull_points[j%len(convex_hull_points)], convex_hull_points[(j+1)%len(convex_hull_points)]) 
    #                 for j in range(len(convex_hull_points))]
    # line_segments = LineCollection(segments = line_to_plot, colors = cmap[i])
    # ax.add_collection(line_segments)
    xx, yy = convex_hull_points.exterior.coords.xy
    x, y = xx.tolist(), yy.tolist()
    line_of_polygon = list(zip(x, y))
    line_to_plot = [(line_of_polygon[j%len(line_of_polygon)], line_of_polygon[(j+1)%len(line_of_polygon)]) 
                    for j in range(len(line_of_polygon))]

    line_segments = LineCollection(segments = line_to_plot)
    ax.add_collection(line_segments)

plt.show()


