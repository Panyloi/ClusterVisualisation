import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

phi = np.linspace(0, 2.*np.pi, 40)
r = 0.5 + np.cos(phi)         # polar coords
x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian


tck, u = splprep([x, y], s=0)
new_points = splev(u, tck)

fig, ax = plt.subplots()
ax.plot(x, y, 'ro')
ax.plot(new_points[0], new_points[1], 'r-')
plt.show()