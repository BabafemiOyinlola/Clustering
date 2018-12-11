import math
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__ (self, k = 3):
        self.k = k
        self.groups = []
        self.iter = 0

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

        self.centroids = final_centroid
        return
        
    def plot(self, title):
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


