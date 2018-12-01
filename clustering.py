import scipy.cluster as sc
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import math
import numpy as np


class HierachicalClustering:
    def __init__(self):
        
        return
    
    def distance(self, data):

        rows = data.shape[0]
        cols = data.shape[1]
        
        distanceMatrix = np.zeros((rows,rows))

        for i in range(rows):
            for j in range(rows):

                sumTotal = 0

                for c in range(cols):

                    sumTotal = sumTotal + pow((data[i,c] - data[j,c]),2)

                distanceMatrix[i,j] = math.sqrt(sumTotal)

        return distanceMatrix

    def calc(self, data):
        condenced_distance = sd.squareform(self.distance(data))
        linkage = sc.hierarchy.linkage(condenced_distance) #link points based on distance
        plt.figure(figsize=(300,100))
        sc.hierarchy.dendrogram(linkage)
        plt.show()




#KMeans using scipy lib
# class KMeans:
#     def __init__(self, k=3):
#         self.k = k
#         self.centroids = []

#     def cluster(self, data):
#         self.centroids, distortion = sc.vq.kmeans(data, self.k)

#     def dist(self, p1, p2):
#         sumTotal = 0

#         for c in range(len(p1)):
#             sumTotal = sumTotal + pow((p1[c] - p2[c]),2)

#         return math.sqrt(sumTotal)


#     def plot(self, data, title):
#         self.cluster(data)
#         plt.figure(figsize=(6,4))
#         centroid_colours = ['mx', 'gx', 'yx', 'rx', 'bx', 'kx', 'wx']
#         plt.plot(data[:,0],data[:,1],'.')
       
#         for i in range(self.k):
#             plt.plot(self.centroids[i][0], self.centroids[0][1],centroid_colours[i])

#         # groups = []
#         # group1 = np.array([])
#         # group2 = np.array([])
#         # for i in range(self.k):
#         #     groups.append(np.array([]))

#         # for d in data:
#         #     if (self.(d, self.centroids[0,:]) < self.dist(d, self.centroids[1,:])):
#         #         if (len(group1) == 0):
#         #             group1 = d
#         #         else:
#         #             group1 = np.vstack((group1,d))
#         #     else:
#         #         if (len(group2) == 0):
#         #             group2 = d
#         #         else:
#         #             group2 = np.vstack((group2,d))

#         # plt.figure(figsize=(6,4))

#         # plt.plot(group1[:,0],group1[:,1],'r.')
#         # plt.plot(group2[:,0],group2[:,1],'g.')

#         # plt.plot(centroids[0,0],centroids[0,1],'rx')
#         # plt.plot(centroids[1,0],centroids[1,1],'gx')
#         # plt.title(title)
#         # plt.show()

