from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

points = np.random.rand(50000, 4)
plt.plot(points[:,0], points[:,1], 'o')

i = 1
while points.shape[0] >= 3:
  print(points.shape)
  hull = ConvexHull(points)
  if points.shape[1] <= 2:
    for s in hull.simplices:
      plt.plot(points[s,0], points[s,1], 'k-')
    plt.savefig("test_%s.png" % i )

  points = np.delete(points, hull.vertices, 0)
  i+=1