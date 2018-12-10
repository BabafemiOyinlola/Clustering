import pandas as pd
import numpy as np
import random
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ClassImbalance:
    def __init__(self):
        return

    def read_data(self, filepath):
        colums = ["Sex","Length","Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", 
                    "Shell_weight", "Class"]
        data = pd.read_csv(filepath, names=colums)
        data = pd.DataFrame(data)

        self.data = np.array(data)
        for i in range(self.data.shape[0]):
            self.data[i, 8] = self.data[i, 8].strip()
    
        return

    #set k = 4174 - 32 = 4142 (positives = 2)
    def pre_process_undersample(self, k, label):

        undersampled_data = np.array([])
        label_index = []

        #select indexes of rows with specified label
        for row in range(self.data.shape[0]):
            if self.data[row, 8] == label:
                label_index.append(row)

        random_remove = random.sample(label_index, k)

        for i in range(self.data.shape[0]):
            if i not in random_remove:
                if len(undersampled_data) == 0:
                    undersampled_data = self.data[i, :].copy()
                else:
                    undersampled_data = np.vstack((undersampled_data, self.data[i, :]))

        return undersampled_data

    def pre_process_oversample(self, k, label):
        oversampled_data = self.data.copy()
        label_index = []

        for row in range(self.data.shape[0]):
            if self.data[row, 8] == label:
                label_index.append(row)

        for i in range(k):
            index = label_index[random.randint(0, (len(label_index)-1))]
            item = self.data[index]

            oversampled_data = np.vstack((oversampled_data, item))
        
        return oversampled_data

    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)):
            total = total + pow(point1[i] + point2[i], 2)

        return math.sqrt(total)

    #Fix this
    def smote(self, k, label):
        data = self.data.copy()
        label_index = []

        for row in range(self.data.shape[0]):
            if self.data[row, 8] == label:
                label_index.append(row)
        
        for i in range(k):
            rand_item = label_index[random.randint(0, (len(label_index) - 1))]

            #KNN
            distances = []
            distance_index = []

            for item in label_index:
                if item == rand_item:
                    continue
                
                distances.append(self.euclidean_distance(self.data[item], self.data[rand_item]))
                distance_index.append(item)

            k_neigh = []

            for neigh in range(2):
                nearest = np.argmin(distances)
                k_neigh.append(distance_index[nearest])
                distances[nearest] = float("inf")
            
            neigh_rand = k_neigh[random.randint(0, (len(k_neigh) - 1))]

            feat = []
            for j in range(7):
                col = self.data[neigh_rand, j] - self.data[rand_item, j]
                feat.append(col)
            
            alpha = random.random()

            all_feat = []

            for j in range(len(feat)):
                col = self.data[rand_item, j] + alpha*feat[j]
                all_feat.append(col)
            
            all_feat.append(label)

            data = np.vstack((data, all_feat))

        return data

    #binary classification
    def logistic_regression_oversampled(self):
        data = self.pre_process_oversample(4110, "positive")

        labels  = data[:, 8]
        features =  np.delete(data, obj=8, axis=1)

        #one hot encode sex
        new_col = pd.get_dummies(features[:, 0])
        #create new columns for sex class
        new_col = np.array(new_col)
        #add the new columns to features
        features = np.column_stack([features, new_col])
        #delete sex column 
        features =  np.delete(features, obj=0, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Accuracy over sampled data: ", accuracy)
        return

    def logistic_regression_undersampled(self):
        data = self.pre_process_undersample(4110, "negative")

        labels  = data[:, 8]
        features =  np.delete(data, obj=8, axis=1)

        #one hot encode sex
        new_col = pd.get_dummies(features[:, 0])
        #create new columns for sex class
        new_col = np.array(new_col)
        #add the new columns to features
        features = np.column_stack([features, new_col])
        #delete sex column 
        features =  np.delete(features, obj=0, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Accuracy under sampled data: ", accuracy)
        return



abalone = ClassImbalance()
abalone.read_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/abalone19.csv")
# undersampled = abalone.pre_process_undersample(4110, "negative") 
# oversampled = abalone.pre_process_oversample(4110, "positive")

# abalone.logistic_regression_oversampled()
abalone.logistic_regression_undersampled()