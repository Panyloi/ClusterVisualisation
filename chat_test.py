import matplotlib.pyplot as plt
import numpy as np

def plot_circle_around_each_point(points, n):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], color='b', label='Points')

    for point in points:
        angles = np.linspace(0, 2*np.pi, n)
        x = point[0] + np.cos(angles)
        y = point[1] + np.sin(angles)
        plt.plot(x, y, color='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Circle with {} points around each point'.format(n))
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Przykładowe punkty o różnych współrzędnych
points = np.array([[1, 2],
                   [3, 4],
                   [5, 6],
                   [7, 8]])

# Liczba punktów na okręgu dla każdego punktu
n = 10

plot_circle_around_each_point(points, n)