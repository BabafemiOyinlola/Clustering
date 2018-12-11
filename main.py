from clustering import *
from classify_mushroom2 import *
from pre_processing import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data_Joensuu()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

#Save plots for data and standardized data
test.plot(dataset, "Latitude", "Longitude", "User Location - JOENSUU")
test.plot(standardised_data, "Latitude", "Longitude", "Standardised Location Data")

# kmeans = KMeans(4)
# kmeans.iterate(standardised_data)
# kmeans.plot("K-means Clustered Data")

#Elbow method
elbow_method = ElbowMethod()
k = range(1, 10)
errors = []

for i in k:
    kmeans = KMeans(i)
    kmeans.iterate(standardised_data)

    sse = elbow_method.sum_of_squared_errors(kmeans.grouped_points, kmeans.centroids)
    errors.append(sse)

plt.plot(k, errors, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


HC = HierachicalClustering(standardised_data)
HC.cluster(standardised_data)
