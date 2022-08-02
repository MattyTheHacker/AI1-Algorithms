import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pandas as pd
import string
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# randomly chosen dataset
X = np.array([[0.4, 0.53], [0.22, 0.38], [0.35, 0.32],
             [0.26, 0.19], [0.08, 0.41], [0.45, 0.3]])
alphabet = [i for i in string.ascii_lowercase]
alphabet = alphabet[0:len(X)]

# drawing distance matrix
df = pd.DataFrame(X, columns=['xcord', 'ycord'], index=alphabet)
asd = pd.DataFrame(distance_matrix(df.values, df.values),
                   index=df.index, columns=df.index)
print(asd)
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean')
cluster.fit_predict(X)
print(cluster.labels_)
print(alphabet)


# drawing table
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:, 0], X[:, 1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

# Drawing Clustering
linked = linkage(X, 'complete')  # choose ur linkage here

labelList = alphabet
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=labelList,
           distance_sort='ascending',
           show_leaf_counts=True)
plt.show()
