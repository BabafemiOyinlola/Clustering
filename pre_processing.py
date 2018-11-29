import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath

    #1) Read Data: This is to read the Joensuu text file specifically
    def read_data(self):
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

    #2) Select features: Select the relevant features from the dataset to be used.
    # Here it is the Latitide and Longitude which we already have

    #3) Standardize. Here use the mean and std dev to detect the global outliers
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
                #apply the standardization formula to each cell: x_std =  (x - col_mean) / col_std
                data_copy[j, i] = (data[j, i] - col_mean) / col_std

        return data_copy
    
    #4) Detect global outliers

    #5) Normalise the standardized data from (-3, 3) to (0, 1)
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
                #apply the normalisation range(0, 1) formula to each cell: x_norm =  (x - col_min) / (col_max - col_min)
                data_copy[j, i] = (data[j, i] - col_min) / (col_max - col_min)
        return data_copy

    #6) Carry out PCA to ortogonalize the components so that they are uncorrelated. 
    # The main purpose of PCA is to reduce the dimensionality of the data set which often has a large number of correlated variables and, 
    # at the same time, to retain as much as possible of the variation present in the data set.

    #cantralise the data before carrying out PCA

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

    # PCA method allows us to construct independent new variables

    #THIS WAS DONE USING SCKITLEARN. IMPLEMENT YOURSELF!!!!
    def PCA(self, data):
        # Observe and analyse the separability of the data and compare with the separability of the original data. 
        # What are the visualisation benefits from using PCA?
        pca = PCA(n_components=2) #we have two components
        pca.fit(data)
        cof = pca.components_
        # print(cof)
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

    #Plot 
    def plot(self, data, x_axis, y_axis, title):
        fig, ax = plt.subplots()
        # fig.set_size_inches(5, 5)
        ax.plot(data[:,0], data[:, 1], ".", color="green", alpha=0.5)
        ax.set_ylabel(y_axis)
        ax.set_xlabel(x_axis)
        ax.set_title(title)
        fig.tight_layout()
       
        plt.show()

    

# test = Preprocessing("Joensuu.txt")
# dataset = test.read_data()
# standardised_data = test.standardize_data(dataset)
# normalised_data = test.normalise_data(standardised_data)
# centralized_data = test.centralise(normalised_data)
# transformed_data = test.PCA(centralized_data)
# test.plot(transformed_data, "Ist PC", "2nd PC", "PCA plot")
# test.percentage_of_variance(centralized_data)

# centroids = normalised_data[np.random.choice(normalised_data.shape[0], 3, False)]
# print("Centroids: ", centroids)

#Correlation between coefficients
#Ideally, if all features are completely independent from each other, the C matrix should be equal to the identity matrix
# print(np.corrcoef(centralized_data, rowvar=False)) #this shows that the features are not completely independent of each other
# test.plot(normalised_data, "Latitude", "Longitude", "Lon & Lat")

# print("Standardized data: ")
# for i in range(len(standardised_data)):
#     print(standardised_data[i])

# print("\n\n")
# print("Normalised data: ")

# for i in range(len(normalised_data)):
#     print(normalised_data[i])
