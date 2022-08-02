import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

points = np.array([[3,3],[4,4],[5,2],[6,7],[7,6],[10,5],[9,3],[8,2]])
print("Dataset shape:", points.shape)

from sklearn.neighbors import NearestNeighbors # importing the library
neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(points) # fitting the data to the object
distances,indices=nbrs.kneighbors(points) # finding the nearest neighbours

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show() # showing the plot

from sklearn.cluster import DBSCAN
# cluster the data into five clusters
dbscan = DBSCAN(eps = 8, min_samples = 4).fit(points) # fitting the model
labels = dbscan.labels_ # getting the labels

# Plot the clusters
plt.scatter(points[:, 0], points[:,1], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("Income") # X-axis label
plt.ylabel("Spending Score") # Y-axis label
plt.show() # showing the plot






