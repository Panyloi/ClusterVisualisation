import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

fig, ax = plt.subplots()


line_segments = LineCollection(segments = (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))))
ax.add_collection(line_segments)
ax.scatter([-2, 2], [-2, 2])

ax.set_title('Line collection with masked arrays')
plt.show()