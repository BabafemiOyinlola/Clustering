import math
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__ (self, k = 3):
        self.k = k
        self.centroids = []
        self.cluster = {}
        self.points = []
        self.point_labels = []
           
    def initial_centroids(self, data):
        #choose initial centroids
        centroids = data[np.random.choice(data.shape[0], self.k, False)]

        return centroids

    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)):
            total = total + pow(point1[i] + point2[i], 2)

        return math.sqrt(total)

    def assign_cluster(self, data):
        #assign the point with the euclidean distance closest to the centroid to that cluster
        self.centroids = self.initial_centroids(data)
        # cluster = {}
        for i in range(data.shape[0]):
            dist_min = np.inf
            label_index = -1

            for j in range(len(self.centroids)):
                dist = self.euclidean_distance(data[i, :], self.centroids[j])

                if(dist < dist_min):
                    dist_min = dist
                    label_index = j

            self.cluster.update({str(data[i, :]): label_index})
        # self.plot(data, "Initial centroids")
        temp = self.color_points(data)
        print("Points: ", temp[0])
        print("Labels: ", temp[1])
        return self.cluster
    
    #make this generic
    def plot(self, data, title):
        plt.figure(figsize=(6,4))
        self.color_points(data)
        for i in range(self.points.shape[0]):
            col = "red"
            if(self.point_labels[i] ==  1.0):
                col = "blue"
            elif(self.point_labels[i] ==  2.0):
                col = "black"
            plt.plot(self.points[i,0],self.points[i,1],'.', color = col)
        plt.plot(self.centroids[0,0], self.centroids[0,1],'mx')
        plt.plot(self.centroids[1,0], self.centroids[1,1],'gx')
        plt.plot(self.centroids[2,0], self.centroids[2,1], 'yx')
        plt.title(title)
        plt.show()

    def color_points(self, data):
        points = np.empty((data.shape[0], 2))
        cluster_labels = np.empty((data.shape[0], 1))
        count = 0
        for key, value in self.cluster.items():
            key = key.replace("]", "")
            key = key.replace("[", "")
            key = key.rstrip()

            lon_lat = key.split(" ")
            i = len(lon_lat) - 1
            while(i >= 0):
                if(lon_lat[i] == "" or lon_lat[i] == " "):
                    lon_lat.pop(i)   
                i = i - 1         

            # print("lat: ",lon_lat[0])
            # print("lon: ", lon_lat[1])
            points[count, 0] = float(lon_lat[0])
            points[count, 1] = float(lon_lat[1])
            # print(value)
            cluster_labels[count] = value
            count = count + 1
        # print(cluster_labels)
        self.points = points
        self.point_labels = cluster_labels
        return(points, cluster_labels)

    def choose_new_centroid(self):
        
