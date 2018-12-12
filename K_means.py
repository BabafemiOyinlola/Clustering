import math
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__ (self, k = 3):
        self.k = k
        self.centroids = []
        # self.cluster = {}
        self.points = []
        self.point_labels = []
        self.grouped_points = []
        self.cluster = []
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

    def assign_cluster(self, data):
        #assign the point with the euclidean distance closest to the centroid to that cluster
        if self.iter == 0:
            self.centroids = self.initial_centroids(data)
        else:
            data = self.points

        for i in range(data.shape[0]):
            dist_min = np.inf
            label_index = -1

            for j in range(len(self.centroids)):
                dist = self.euclidean_distance(data[i, :], self.centroids[j])

                if(dist < dist_min):
                    dist_min = dist
                    label_index = j
            
            self.cluster.append(str(data[i, :]) + ":" + str(label_index))


        return self.cluster
  
    #make this generic
    def plot(self, data, title):
        plt.figure(figsize=(6,4))
        self.label_points(data)
        col = ""
        for i in range(self.points.shape[0]):
            if(self.point_labels[i] ==  0.0):
                col = "blue"
            elif(self.point_labels[i] ==  1.0):
                col = "red"
            elif(self.point_labels[i] ==  2.0):
                col = "black"
            elif(self.point_labels[i] ==  3.0):
                col = "brown"
            elif(self.point_labels[i] ==  4.0):
                col = "green"
            elif(self.point_labels[i] ==  5.0):
                col = "cyan"

            plt.plot(self.points[i,0],self.points[i,1],'.', color = col)
      
        centroid_colours = ['mx', 'gx', 'yx', 'rx', 'bx', 'kx', 'wx']
       
        for i in range(self.k):
            plt.plot(self.centroids[i][0], self.centroids[0][1],centroid_colours[i])
        plt.title(title)
        plt.show()

    def label_points(self, data):
        if self.iter == 0:
            self.centroids = self.initial_centroids(data)
        else:
            data = self.points

        points = np.empty((data.shape[0], 2))
        cluster_labels = np.empty((data.shape[0], 1))

        point_in_clusters = [[] for _ in range(self.k)]  

        for i in range(data.shape[0]):
            item = self.cluster[i].split(":")
            
            temp = item[0].replace("]", "")
            temp = temp.replace("[", "")
            temp = temp.rstrip()

            points[i] = np.fromstring(temp, dtype=float, sep=' ')
            # print(points[i])
            cluster_labels[i] = item[1]
        
            for j in range(self.k):
                if(item[1] == str(j)):
                    point_in_clusters[j].append(points[i])

        # total = len(point_in_clusters[0]) + len(point_in_clusters[1]) + len(point_in_clusters[2]) + len(point_in_clusters[3])
        self.grouped_points = point_in_clusters
        self.points = points
        self.point_labels = cluster_labels

        return(points, cluster_labels)

    def choose_new_centroid(self):
        #choose new centroids
        new_centroids = np.empty((self.k, 2)) 

        total_x = 0
        total_y = 0

        for i in range(len(self.grouped_points)):
            temp = self.grouped_points[i]

            for j in range(len(temp)):
                total_x = total_x + temp[j][0]
                total_y = total_y + temp[j][1]

            new_centroids[i][0] = total_x/len(temp)
            new_centroids[i][1] = total_y/len(temp)

        dist_btw_centroids = [0]*self.k

        for i in range(self.k):
            dist_btw_centroids[i] = self.euclidean_distance(new_centroids[i], self.centroids[i])

        total_diff = sum(dist_btw_centroids)
        print("Old Centroid: ")
        print(self.centroids)
        print("New centroid: ")
        print(new_centroids)

        self.centroids = new_centroids

        return (total_diff)

    def iterate(self, data, threshold=0.001):
        self.assign_cluster(data)
        self.label_points(data)

        diff_inital = sum(self.centroids)
        print("Difference initial: ", diff_inital)

        diff = diff_inital


        count = 10
        # while diff > threshold:
        while count > 0:
            self.iter = self.iter + 1
            self.choose_new_centroid() 
            self.assign_cluster(data) 
            self.label_points(data)      
            # print("Difference before: ", diff)
            # diff_new = sum(self.centroids)
            # print("Difference after: ", diff_new)
            # diff = diff_new - diff
            # print("Difference: ", diff)
            # count = count - 1

        return(self.points)
