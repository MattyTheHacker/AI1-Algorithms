import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import string
from scipy.cluster.hierarchy import dendrogram, linkage


alphabet = [i for i in string.ascii_lowercase]

Distance_Matrix = [
    [0.0, 0.2, 0.15, 0.76, 0.54, 0.31],
    [0.2, 0.0, 0.89, 0.18, 0.66, 0.27],
    [0.15, 0.89, 0.0, 0.82, 0.73, 0.56],
    [0.76, 0.18, 0.82, 0.0, 0.42, 0.39],
    [0.54, 0.66, 0.73, 0.42, 0.0, 0.51],
    [0.31, 0.27, 0.56, 0.39, 0.51, 0.0],
]
alphabet = alphabet[:len(Distance_Matrix)]
distArray = ssd.squareform(Distance_Matrix)

# Drawing Clustering
linked = linkage(distArray, 'single')  # choose ur linkage here

labelList = alphabet
plt.figure(figsize=(10, 7))
dendro = dendrogram(linked,
                    orientation='top',
                    labels=labelList,
                    distance_sort='ascending',
                    show_leaf_counts=True)
finalHeight = [y[1] for y in dendro['dcoord']]
print(sorted(finalHeight))
plt.show()
