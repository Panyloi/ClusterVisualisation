import numpy as np
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
