import numpy as np
from sklearn.cluster import DBSCAN

X = np.array([[3, 3], [4, 4], [5, 2], [6, 7], [7, 6], [10, 5], [9, 3], [8, 2]])

clustering = DBSCAN(eps=2.5, min_samples=2).fit(X)
print(clustering.labels_)
