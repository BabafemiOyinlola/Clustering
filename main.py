from pre_processing import *
from K_means import *

test = Preprocessing("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/Joensuu.txt")
dataset = test.read_data()
standardised_data = test.standardize_data(dataset)
normalised_data = test.normalise_data(standardised_data)
centralized_data = test.centralise(normalised_data)
transformed_data = test.PCA(centralized_data)

# centroids = normalised_data[np.random.choice(normalised_data.shape[0], 3, False)]
# print("Centroids: ", centroids)


kmeans = KMeans(3)
clusters = kmeans.assign_cluster(standardised_data)
clustered_points = kmeans.iterate(standardised_data)
# clusters2 = kmeans.assign_cluster(normalised_data)

kmeans.plot(clustered_points, "Clustered data")
print(clusters)
# for i in range(len(clusters)):
#     print(clusters[i])
