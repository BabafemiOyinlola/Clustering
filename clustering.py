<<<<<<< Updated upstream
=======
import math
import time
import numpy as np
>>>>>>> Stashed changes
import scipy.cluster as sc
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import math
import numpy as np


class HierachicalClustering:
<<<<<<< Updated upstream
    def __init__(self):
        
        return
=======
    def __init__(self, data):
        self.data = data
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    
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

<<<<<<< Updated upstream
    def calc(self, data):
        condenced_distance = sd.squareform(self.distance(data))
        linkage = sc.hierarchy.linkage(condenced_distance) #link points based on distance
        plt.figure(figsize=(300,100))
        sc.hierarchy.dendrogram(linkage)
        plt.show()



=======
    def calc(self):
        distance = self.distance()
        
        #Calculate time:
        start_time = time.process_time() #start time

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
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

        end_time = time.process_time() - start_time 
        print("Time for Hierarchical Clustering: ", end_time)

        plt.show()

        return


class KMeans:

    def __init__ (self, k = 3):
        try:
            if k > 5 :
                self.k = None
                raise Exception("Maximum clusters is 5")
            elif k < 1:
                self.k = None
                raise Exception("Minimum clusters is 2")
            else:
                self.k = k
                self.groups = []
                self.iter = 0
        except Exception as error:
            print(error)

    def initial_centroids(self, data):
        #choose initial centroids
        centroids = data[np.random.choice(data.shape[0], self.k, False)]
        return centroids

    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)):
            total = total + pow(point1[i] + point2[i], 2)

        return math.sqrt(total)

    def cluster(self, data, cent):
        centroids = cent
        self.centroids = centroids

        grouped_points = [[] for _ in range(self.k)]  

        for i in range(data.shape[0]):
            dist_min = np.inf
            label_index = -1

            for j in range(self.k):
                dist = self.euclidean_distance(data[i, :], centroids[j])
                if(dist < dist_min):
                    dist_min = dist
                    label_index = j

            grouped_points[label_index].append(data[i, :])
        self.grouped_points = grouped_points
        
        return

<<<<<<< Updated upstream
<<<<<<< Updated upstream
#         # plt.plot(centroids[0,0],centroids[0,1],'rx')
#         # plt.plot(centroids[1,0],centroids[1,1],'gx')
#         # plt.title(title)
#         # plt.show()

=======
=======
>>>>>>> Stashed changes
    def new_centroid(self):
        new_centroids = np.empty((self.k, 2))  
        points = self.grouped_points

        for i in range(self.k):
            total_x = 0
            total_y = 0

            temp = points[i]
            for j in range(len(points[i])):
                total_x = total_x + temp[j][0]
                total_y = total_y + temp[j][1]

            new_centroids[i][0] = total_x /len(temp)
            new_centroids[i][1] = total_y /len(temp)

        return new_centroids

    def centroid_diff(self, old_centroids, new_centroids):
        dist = [0]*self.k

        for i in range(self.k):
            dist[i] = self.euclidean_distance(new_centroids[i], old_centroids[i])


        difference = sum(dist)

        return difference

    def iterate(self, data):
        try:
            if self.k == None:
                raise Exception("")
                return
            else:
                #Calculate time:
                start_time = time.process_time() #start time

                centroid = self.initial_centroids(data)
                self.cluster(data, centroid)

                diff = 9999999999
                threshold = 0.0001

                count = 0
                temp = 0
                final_centroid = centroid

                while(diff > threshold):
                    new_centroid = self.new_centroid()
                    diff_new = self.centroid_diff(centroid, new_centroid)
                    self.cluster(data, new_centroid)
                    
                    if count > 0:
                        diff = abs(diff_new - temp)

                    print("Old centroid: ")
                    print(centroid)
                    print("New centroid: ")
                    print(new_centroid)
                    print("Difference: ", diff)

                    centroid = new_centroid
                    temp = diff_new
                    count = count + 1
                    final_centroid = new_centroid

                end_time = time.process_time() - start_time 
                print("Time for K-Means Clustering: ", end_time)

                self.centroids = final_centroid

                return
        except Exception as error:
            print(error)
            
    def plot(self, title):
        try:
            if self.k == None:
                raise Exception("")
                return
            else:
                plt.figure(figsize=(6,4))
                col = ["blue", "red", "black", "brown", "green", "cyan"]
                points = self.grouped_points
                for i in range(len(points)):
                    temp = points[i]
                    for j in range(len(temp)):
                        plt.plot(temp[j][0],temp[j][1],'.', color = col[i])
            
                centroid_colours = ['mx', 'gx', 'yx', 'rx', 'bx', 'kx', 'wx']
                print("FINAL CENTROIDS")
                print(self.centroids)
                for i in range(self.k):
                    centroid = self.centroids[i]
                    plt.plot(centroid[0],centroid[1], centroid_colours[i])
                # plt.scatter(self.centroids[:,0] ,self.centroids[:,1], color="green")
                plt.title(title)
                plt.show()

                return
        except Exception as error:
            print(error)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
