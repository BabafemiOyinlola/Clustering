import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from pre_processing import Preprocessing


class ClassImbalance:
    # def __init__(self):
    def read_data(self, filepath):
        colums = ["Sex","Length","Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", 
                    "Shell_weight", "Class"]
        data = pd.read_csv(filepath, names=colums)
        data = pd.DataFrame(data)

        self.data = np.array(data)
        for i in range(self.data.shape[0]):
            self.data[i, 8] = self.data[i, 8].strip()
        
        return self.data
    
    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)):
            total = total + pow(point1[i] + point2[i], 2)

        return math.sqrt(total)

    def pre_process_undersample(self, k, label, data):
        undersampled_data = np.array([])
        label_index = []
        
        #select indexes of rows with specified label
        for row in range(data.shape[0]):
            if data[row, 8] == label:
                label_index.append(row)

        random_remove = random.sample(label_index, k)

        for i in range(data.shape[0]):
            if i not in random_remove:
                if len(undersampled_data) == 0:
                    undersampled_data = data[i, :].copy()
                else:
                    undersampled_data = np.vstack((undersampled_data, data[i, :]))

        return undersampled_data

    def pre_process_oversample(self, k, label, data):
        oversampled_data = data.copy()
        label_index = []

        for row in range(data.shape[0]):
            if data[row, 8] == label:
                label_index.append(row)

        for i in range(k):
            index = label_index[random.randint(0, (len(label_index)-1))]
            item = data[index]

            oversampled_data = np.vstack((oversampled_data, item))
        return oversampled_data
    
    def smote(self, x_train, y_train):
        sm = SMOTE(random_state=12, ratio = 1.0)
        features_train_smote, labels_train_smote = sm.fit_sample(x_train, y_train) 
        return (features_train_smote, labels_train_smote)

    #binary classification
    def logistic_regression_oversampled(self):
        train, test = self.process_and_split_data()

        train_oversampled = self.pre_process_oversample(1219, "positive", train)

        x_train = np.delete(train_oversampled, obj=8, axis=1)
        y_train = train_oversampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        reg = LogisticRegression()
        reg.fit(features_train, y_train)
        pred = reg.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy over sampled data without PCA: ", accuracy)
        print()
        self.metrics(pred, y_test)
        # self.roc_curve_acc(y_test, pred, "Logisitic Regression oversampled data without PCA:")
        print()
        
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
       
        return cross_val_acc, y_test, pred

    def logistic_regression_undersampled(self):
        train, test = self.process_and_split_data()

        train_undersampled = self.pre_process_undersample(1219, "negative", train)

        x_train = np.delete(train_undersampled, obj=8, axis=1)
        y_train = train_undersampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new encoded columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        reg = LogisticRegression()
        reg.fit(features_train, y_train)
        pred = reg.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy under sampled data without PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
       
        return cross_val_acc, y_test, pred

    def logistic_regression_smote(self):

        train, test = self.process_and_split_data()

        x_train = np.delete(train, obj=8, axis=1)
        y_train = train[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #Handle imbalance
        features_train, y_train = self.smote(features_train, y_train)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        reg = LogisticRegression()
        reg.fit(features_train, y_train)
        pred = reg.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy over smote without PCA: ", accuracy)
        print()
        self.metrics(pred, y_test)
        print()
        
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
       
        return cross_val_acc, y_test, pred

    def logistic_regression_smote_PCA(self):

        train, test = self.process_and_split_data()

        x_train = np.delete(train, obj=8, axis=1)
        y_train = train[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)
        
        #Handle imbalance
        features_train, y_train = self.smote(features_train, y_train)
        
        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        reg = LogisticRegression()
        reg.fit(features_train, y_train)
        pred = reg.predict(features_test)

        #PCA
        features_train = self.PCA(features_train, 5)
        features_test = self.PCA(features_test, 5)
        variance = self.percentage_of_variance(features_train, "Abalone", 5) 

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy smote with PCA: ", accuracy)
        print()
        self.metrics(pred, y_test)
        print()
        
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
       
        return cross_val_acc, y_test, pred

    def logistic_regression_oversampled_PCA(self):
        train, test = self.process_and_split_data()

        train_oversampled = self.pre_process_oversample(1219, "positive", train)

        x_train = np.delete(train_oversampled, obj=8, axis=1)
        y_train = train_oversampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        #PCA
        features_train = self.PCA(features_train, 5)
        features_test = self.PCA(features_test, 5)
        variance = self.percentage_of_variance(features_train, "Abalone", 5) 

        reg = LogisticRegression()
        reg.fit(features_train, y_train)
        pred = reg.predict(features_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy over sampled data after PCA: ", accuracy)
        print()
        
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)

        return cross_val_acc, y_test, pred

    def logistic_regression_undersampled_PCA(self):
        train, test = self.process_and_split_data()

        train_undersampled = self.pre_process_undersample(1219, "negative", train)

        x_train = np.delete(train_undersampled, obj=8, axis=1)
        y_train = train_undersampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new encoded columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        #PCA
        features_train = self.PCA(features_train, 5)
        features_test = self.PCA(features_test, 5)

        variance = self.percentage_of_variance(features_train, "Abalone", 5) 

        reg = LogisticRegression()
        reg.fit(features_train, y_train)
        pred = reg.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy under sampled data after PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
        return cross_val_acc, y_test, pred

    def decision_tree_oversampled(self):
        train, test = self.process_and_split_data()

        train_oversampled = self.pre_process_oversample(1219, "positive", train)

        x_train = np.delete(train_oversampled, obj=8, axis=1)
        y_train = train_oversampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test

        tree = DecisionTreeClassifier()
        tree.fit(features_train, y_train)
        pred = tree.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy over sampled data without PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_undersampled(self):
        train, test = self.process_and_split_data()

        train_undersampled = self.pre_process_undersample(1219, "negative", train)

        x_train = np.delete(train_undersampled, obj=8, axis=1)
        y_train = train_undersampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new encoded columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        tree = DecisionTreeClassifier()
        tree.fit(features_train, y_train)
        pred = tree.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy under sampled data without PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred
    
    def decision_tree_smote(self):
        train, test = self.process_and_split_data()

        x_train = np.delete(train, obj=8, axis=1)
        y_train = train[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #Handle imbalance
        features_train, y_train = self.smote(features_train, y_train)
        
        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test

        tree = DecisionTreeClassifier()
        tree.fit(features_train, y_train)
        pred = tree.predict(features_test)
        
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(features_train, y_train)
        pred = knn.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy smote data without PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_smote_PCA(self):
        train, test = self.process_and_split_data()

        x_train = np.delete(train, obj=8, axis=1)
        y_train = train[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #Handle imbalance
        features_train, y_train = self.smote(features_train, y_train)
        
        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test

        #PCA
        features_train = self.PCA(features_train, 5)
        features_test = self.PCA(features_test, 5)
        variance = self.percentage_of_variance(features_train, "Abalone", 5) 

        tree = DecisionTreeClassifier()
        tree.fit(features_train, y_train)
        pred = tree.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy smote with PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_oversampled_PCA(self):
        train, test = self.process_and_split_data()

        train_oversampled = self.pre_process_oversample(1219, "positive", train)

        x_train = np.delete(train_oversampled, obj=8, axis=1)
        y_train = train_oversampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test

         #PCA
        features_train = self.PCA(features_train, 5)
        features_test = self.PCA(features_test, 5)
        variance = self.percentage_of_variance(features_train, "Abalone", 5) 

        tree = DecisionTreeClassifier()
        tree.fit(features_train, y_train)
        pred = tree.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy over sampled data after PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_undersampled_PCA(self):
        train, test = self.process_and_split_data()

        train_undersampled = self.pre_process_undersample(1219, "negative", train)

        x_train = np.delete(train_undersampled, obj=8, axis=1)
        y_train = train_undersampled[:, 8]
        x_test = np.delete(test, obj=8, axis=1)
        y_test = test[:, 8]

        new_col = pd.get_dummies(x_train[:, 0])
        new_col2 = pd.get_dummies(x_test[:, 0])

        #create new encoded columns for sex class
        new_col = np.array(new_col)
        new_col2 = np.array(new_col2)
        #add the new columns to features
        features_train = np.column_stack([x_train, new_col])
        features_test = np.column_stack([x_test, new_col2])

        #delete sex column 
        features_train =  np.delete(features_train, obj=0, axis=1)
        features_test =  np.delete(features_test, obj=0, axis=1)

        #standardize data
        preprocess = Preprocessing()
        features_train = preprocess.standardize_data(features_train)
        features_test = preprocess.standardize_data(features_test)

        #PCA
        features_train = self.PCA(features_train, 5)
        features_test = self.PCA(features_test, 5)
        variance = self.percentage_of_variance(features_train, "Abalone", 5) 

        tree = DecisionTreeClassifier()
        tree.fit(features_train, y_train)
        pred = tree.predict(features_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy under sampled data after PCA: ", accuracy)
        print()
        features = np.vstack((features_train, features_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def plot_imbalance(self):
        data = self.data

        positive = 0
        negative = 0

        for i in range(data.shape[0]):
            if data[i, 8] == "negative":
                negative += 1
            elif data[i, 8] == "positive":
                positive += 1
        classes = ["negative", "positive"]
        class_count = [negative, positive]

        barplot = plt.bar(classes, class_count)
        barplot[1].set_color("yellow")
        plt.xlabel("Classes")
        plt.ylabel("Count")
        plt.title("Class Imbalance")  
        plt.savefig("Class Imbalance.jpeg")
        plt.show()

        return

    def PCA(self, data, n):
        pca = PCA(n_components=n)  #10 features (excluding sex and including encoded sex classes)
        pca.fit(data)
        cof = pca.components_
        trasform_data = pca.transform(data)
        return trasform_data

    def percentage_of_variance(self, data, name, n):
        pca = PCA(n_components=n)
        pca.fit(data)
        colums = ["Length","Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", 
                    "Shell_weight", "Sex-F", "Sex-I", "Sex-M"]
        # plt.bar(colums, pca.explained_variance_ratio_, tick_label= colums)
        plt.plot(colums[0:n], pca.explained_variance_ratio_)
        plt.xlabel("Principal Component")
        plt.ylabel("% Variance Explained")
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.title("Percentage of Variance")   
        plt.savefig("Percentage of variance " + name + ".jpeg")
        # plt.show()
        return pca.explained_variance_ratio_

    def process_and_split_data(self):
        data = self.data
        positive_class = []
        negative_class = []

        train, test = [], []

        for row in range(data.shape[0]):
            if data[row, 8] == "negative":
                negative_class.append(data[row])
            else:
                positive_class.append(data[row])

        positive_class = np.array(positive_class)
        negative_class = np.array(negative_class)
        
        #split in 30% test and 70% train
        pos_num = int(0.3 * len(positive_class))
        neg_num = int(0.3 * len(negative_class))

        for i in range(pos_num):
            index = random.randint(0, (len(positive_class)-1))
            item = positive_class[index]
            test.append(item)         
            positive_class = np.delete(positive_class, obj=index, axis=0)
        
        for i in range(neg_num):
            index = random.randint(0, (len(negative_class)-1))
            item = negative_class[index]
            test.append(item)         
            negative_class = np.delete(negative_class, obj=index, axis=0)

        test = np.array(test)

        for i in range(len(positive_class)):
            train.append(positive_class[i])
        for i in range(len(negative_class)):
            train.append(negative_class[i])
        
        train = np.array(train)

        return (train, test)

    # def process_split_PCA(self, data):
    #     labels  = data[:, 8]
    #     features =  np.delete(data, obj=8, axis=1)

    #     #one hot encode sex
    #     new_col = pd.get_dummies(features[:, 0])
    #     #create new columns for sex class
    #     new_col = np.array(new_col)
    #     #add the new columns to features
    #     features = np.column_stack([features, new_col])
    #     #delete sex column 
    #     features =  np.delete(features, obj=0, axis=1)

    #     features = self.PCA(features, 5)
    #     variance = self.percentage_of_variance(features, "Abalone", 5) 
    #     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    #     return (x_train, x_test, y_train, y_test)

    def metrics(self, predictions, true_labels, name=""):
        accuracy = metrics.accuracy_score(true_labels, predictions)
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        # recall = metrics.recall_score(true_labels, predictions)
        # precision = metrics.pr
        print("Accuracy: ",accuracy)
        print("Confusion matrix: ", confusion_matrix)
        # print("Recall: ",recall)
        # self.roc_curve_acc(true_labels, predictions, name)
        return

    def cross_validation(self,algorithm, features, labels):
        accuracy = cross_val_score(algorithm, features, labels, scoring='accuracy', cv = 10)
        accuracy_percent = accuracy.mean() * 100
        print("Cross validation accuracy: " , accuracy_percent)
        return accuracy

    def plot_metrics(self):
        lg_over = self.logistic_regression_oversampled()
        lg_over_PCA = self.logistic_regression_oversampled_PCA()
        lg_under = self.logistic_regression_undersampled()
        lg_under_PCA = self.logistic_regression_undersampled_PCA()
        lg_SMOTE = self.logistic_regression_smote()
        lg_SMOTE_PCA = self.logistic_regression_smote_PCA()

        dc_over = self.decision_tree_oversampled()
        dc_over_PCA = self.decision_tree_oversampled_PCA()
        dc_under = self.decision_tree_undersampled()
        dc_under_PCA = self.decision_tree_undersampled_PCA()
        dc_SMOTE = self.decision_tree_smote()
        dc_SMOTE_PCA = self.decision_tree_smote_PCA()

        accuracies = [lg_over[0],lg_under[0], lg_SMOTE[0], dc_over[0], dc_under[0], dc_SMOTE[0]]
        accuracies_labels = ["lg_over","lg_under", "lg_SMOTE", "dtc_over", "dtc_under", "dtc_SMOTE"]

        accuracies_PCA = [lg_under_PCA[0], lg_over_PCA[0], lg_SMOTE_PCA[0], dc_over_PCA[0], dc_under_PCA[0], dc_over_PCA[0]]
        accuracies_labels_PCA = ["lg_under_PCA", "lg_over_PCA", "lg_SMOTE_PCA", "dtc_over_PCA", "dtc_under_PCA", "dtc_SMOTE_PCA"]

        means, means_PCA = [], []
        std_dev, std_dev_PCA = [], []
        x_axis = np.arange(len(accuracies))

        for i in range(len(accuracies)):
            temp = np.array(accuracies[i])
            mean = np.mean(temp)
            std = np.std(temp)
            means.append(mean)
            std_dev.append(std)

        for i in range(len(accuracies_PCA)):
            temp = np.array(accuracies_PCA[i])
            mean = np.mean(temp)
            std = np.std(temp)
            means_PCA.append(mean)
            std_dev_PCA.append(std)

       
        width = 0.25
        fig, ax = plt.subplots()
        plt1 = ax.bar(x_axis, means, width,  yerr=std_dev, align='center', alpha=0.5, ecolor='black', capsize=10)
        plt2 = ax.bar(x_axis+width, means_PCA, width, yerr=std_dev_PCA, align='center', alpha=0.5, ecolor='black', capsize=10, color="darkblue")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(accuracies_labels)
        ax.set_title("Average model accuracy and error")
        ax.yaxis.grid(True)
        ax.legend((plt1[0], plt2[0]), ("Before PCA", "After PCA"))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("Model accuracy for Abalone - Class Imbalance")
        plt.show()


        true_labels = [lg_over[1], lg_under[1], lg_SMOTE[1], dc_over[1], dc_under[1], dc_SMOTE[1]]
        predictions = [lg_over[2],lg_under[2], lg_SMOTE_PCA[2], dc_over[2], dc_under[2], dc_SMOTE_PCA[2]]
        col1 = ["yellow", "m", "grey", "pink", "salmon", "cadetblue"]
        col2 = ["blue", "red", "black", "brown", "green", "cyan"]
        accuracies_labels = ["lg_over","lg_under", "lg_SMOTE", "dtc_over", "dtc_under", "dtc_SMOTE"]

        encoder = LabelEncoder()

        for i in range(len(true_labels)):
            true_labels_new = encoder.fit_transform(true_labels[i])
            predictions_new = encoder.fit_transform(predictions[i])
            false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(true_labels_new,predictions_new, pos_label=1)
            roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        
            #plt.figure()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.plot([0, 1], [0, 1], color=col1[i], linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.plot(false_positive_rate, true_positive_rate, color=col2[i], lw=2, label= accuracies_labels[i] + " area = %0.2f)" % roc_auc)
        plt.title("ROC Curve showing various classifiers")
        plt.legend(loc="lower right")
        plt.savefig("ROC Curve multiple curves - Abalone.jpeg")
        plt.show()

        return


abalone = ClassImbalance()
data = abalone.read_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/abalone19.txt")

# # abalone.plot_imbalance()

# abalone.logistic_regression_smote()
# abalone.logistic_regression_smote_PCA()
# abalone.logistic_regression_oversampled()
# abalone.logistic_regression_oversampled_PCA()
# abalone.logistic_regression_undersampled()
# abalone.logistic_regression_undersampled_PCA()
# abalone.decision_tree_smote()
# abalone.decision_tree_smote_PCA()
# abalone.decision_tree_oversampled()
# abalone.decision_tree_oversampled_PCA()
# abalone.decision_tree_undersampled()
# abalone.decision_tree_undersampled_PCA()

abalone.plot_metrics()
