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
        self.grouped_points = []
           
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

        for i in range(data.shape[0]):
            dist_min = np.inf
            label_index = -1

            for j in range(len(self.centroids)):
                dist = self.euclidean_distance(data[i, :], self.centroids[j])

                if(dist < dist_min):
                    dist_min = dist
                    label_index = j

            # self.cluster.update({str(data[i, :]): label_index})
            self.cluster[str(data[i, :])] = label_index
        return self.cluster
    
    #make this generic
    def plot(self, data, title):
        plt.figure(figsize=(6,4))
        self.label_points(data)
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

    def label_points(self, data):
        points = np.empty((data.shape[0], 2))
        cluster_labels = np.empty((data.shape[0], 1))
        count = 0

        point_in_clusters = [[] for _ in range(self.k)]  
        
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

            points[count, 0] = float(lon_lat[0])
            points[count, 1] = float(lon_lat[1])

            cluster_labels[count] = value

            for i in range(self.k):
                if(value == i):
                    point_in_clusters[i].append(points[count])
                    break

            count = count + 1

        self.grouped_points = point_in_clusters
        # print("ALL: ", str(len(point_in_clusters[0]) + len(point_in_clusters[1]) + len(point_in_clusters[2])))

        # print(cluster_labels)
        self.points = points
        self.point_labels = cluster_labels
        
        return(points, cluster_labels)

    def choose_new_centroid(self):
        #choose new centroids
        new_centroids = [[] for _ in range(self.k)] 

        total_x = 0
        total_y = 0

        for i in range(len(self.grouped_points)):
            temp = self.grouped_points[i]

            for j in range(len(temp)):
                total_x = total_x + temp[j][0]
                total_y = total_y + temp[j][1]

            new_centroids[i] = [(total_x/len(temp)), (total_y)/len(temp)]

        diff_btw_centroids = new_centroids - self.centroids

        self.centroids = new_centroids

        return (new_centroids, diff_btw_centroids)

    def iterate(self, data, threshold=0.001):
        self.assign_cluster(data)
        self.label_points(data)
        diff = self.choose_new_centroid()
        while(max(diff) > 0.001):
            
            diff = self.choose_new_centroid()
        return(self.points)


