import math
import timeit
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster as sc
import scipy.spatial.distance as sd


class HierachicalClustering:
    def __init__(self, data):
        self.data = data
        return
    
    def distance(self):
        rows = self.data.shape[0]
        cols = self.data.shape[1]
        
        distanceMatrix = np.zeros((rows,rows))

        for i in range(rows):
            for j in range(rows):
                sumTotal = 0
                for c in range(cols):
                    sumTotal = sumTotal + pow((self.data[i,c] - self.data[j,c]),2)
                distanceMatrix[i,j] = math.sqrt(sumTotal)

        return distanceMatrix

    def calc(self):
        distance = self.distance()
        condenced_distance = sd.squareform(distance)
        linkage = sc.hierarchy.linkage(condenced_distance) #link points based on distance
        # sc.hierarchy.dendrogram(linkage)  
        # plt.savefig("Location dendrogram without pruning")      
        #prune the tree here
        print("Prunning tree")
        # sc.hierarchy.dendrogram(linkage, truncate_mode="lastp", p =0.2)
        # plt.savefig("Location dendrogram prunned at p=0.25")  
        sc.hierarchy.dendrogram(linkage, truncate_mode="level", p = 3)
        plt.savefig("Location dendrogram prunned at level p=3") 
        # sc.hierarchy.dendrogram(linkage, truncate_mode="lastp", p =0.1)
        # plt.savefig("Location dendrogram prunned at p=0.15")
        plt.show()
        # %timeit calc()
        return