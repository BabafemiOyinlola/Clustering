import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Preprocessing:
    def __init__(self, filepath=""):
        self.filepath = filepath

    def read_data_Joensuu(self):
        read = open(self.filepath, "r")
        content = read.readlines()
        data_array = np.empty((len(content), 2)) #Create an empty array to store the lat and lon
        if(len(content) == 0):
            return
        else:
            for line in range(len(content)):
                lat, lon = str(content[line].rstrip()).split(" ")
                content[line] = lat + ","  + lon
                data_array[line, 0] = float(lat)
                data_array[line, 1] = float(lon)
            return data_array
        read.close()

    def standardize_data(self, data):
        data_copy = data.copy()

        rows = data.shape[0]
        cols = data.shape[1]
        #do this for each column in the matrix
        for i in range(cols):
            col_mean = np.mean(data[:, i])
            col_std = np.std(data[:, i])

            #Iterate through the rows in that column
            for j in range(rows):
                data_copy[j, i] = (data[j, i] - col_mean) / col_std

        return data_copy 

    def normalise_data(self, data):
        data_copy = data.copy()

        rows = data.shape[0]
        cols = data.shape[1]
        #do this for each column in the matrix
        for i in range(cols):
            col_max = np.amax(data[:, i])
            col_min = np.amin(data[:, i])

            #Iterate through the rows in that column
            for j in range(rows):
                data_copy[j, i] = (data[j, i] - col_min) / (col_max - col_min)
        return data_copy

    def centralise(self, data):
        data_copy = data.copy()

        rows = data.shape[0]
        cols = data.shape[1]
        #do this for each column in the matrix
        for i in range(cols):
            col_mean= np.mean(data[:, i])

            #Iterate through the rows in that column
            for j in range(rows):
                #apply the normalisation range(0, 1) formula to each cell: x_cen =  (x - col_mean)
                data_copy[j, i] = data[j, i] - col_mean

        return data_copy

    def PCA(self, data, n=2):
        pca = PCA(n_components=n)
        pca.fit(data)
        cof = pca.components_
        trasform_data = pca.transform(data)

        return trasform_data
    
    def percentage_of_variance(self, data):
        pca = PCA(n_components=2) #we have two components
        pca.fit(data)
        # cof = pca.components_
        plt.bar([1, 2], pca.explained_variance_ratio_, tick_label= [1, 2])
        plt.xlabel("Principal Component")
        plt.ylabel("% Variance Explained")

        plt.show()

    def plot(self, data, x_label, y_label, title):
        fig, ax = plt.subplots()
        # fig.set_size_inches(5, 5)
        ax.plot(data[:,0], data[:, 1], ".", color="green")
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        fig.tight_layout()   
        plt.savefig(title + ".jpeg" ,bbox_inches= "tight")
        plt.show()
        
    def read_mushroom_data(self):
        read = open(self.filepath, "r")
        content = read.readlines()
        if(len(content) == 0):
            return
        else:
            len_content = len(content[0].rstrip().split(",")) #Split the first line in content to obtain number of columns/ features
            data_array = np.empty((len(content), len_content), dtype=str) #Create an empty array to store dataitems
            for line in range(len(content)):
                data_array[line] = content[line].split(",")

            read.close()
            labels = data_array[:, 0]
            features =  np.delete(data_array, obj=0, axis=1)
            data_array = []
            return (features, labels)
