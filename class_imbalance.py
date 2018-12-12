import random
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class ClassImbalance:
    def __init__(self):
        # self.x_train x_test, y_train, y_test
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

        x_train, x_test, y_train, y_test = self.process_and_split_data(data)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy over sampled data without PCA: ", accuracy)
        print()
        self.metrics(pred, y_test)
        # self.roc_curve_acc(y_test, pred, "Logisitic Regression oversampled data without PCA:")
        print()
        
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
       
        return cross_val_acc, y_test, pred

    def logistic_regression_undersampled(self):
        data = self.pre_process_undersample(4110, "negative")

        x_train, x_test, y_train, y_test = self.process_and_split_data(data)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy under sampled data without PCA: ", accuracy)
        print()
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)

        # self.roc_curve_acc(y_test, pred, "Logisitic Regression undersampled data without PCA:")
       
        return cross_val_acc, y_test, pred

    def logistic_regression_oversampled_PCA(self):
        data = self.pre_process_oversample(4110, "positive")
        
        x_train, x_test, y_train, y_test = self.process_split_PCA(data)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy over sampled data after PCA: ", accuracy)
        print()
        
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
        # self.roc_curve_acc(y_test, pred, "Logisitic Regression oversampled data - PCA:")

        return cross_val_acc, y_test, pred

    def logistic_regression_undersampled_PCA(self):
        data = self.pre_process_undersample(4110, "negative")
        
        x_train, x_test, y_train, y_test = self.process_split_PCA(data)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Logisitic Regression - Accuracy under sampled data after PCA: ", accuracy)
        print()
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(reg, features, labels)
        # self.roc_curve_acc(y_test, pred, "Logisitic Regression undersampled data with PCA:")
        return cross_val_acc, y_test, pred

    def decision_tree_oversampled(self):
        data = self.pre_process_oversample(4110, "positive")

        x_train, x_test, y_train, y_test = self.process_and_split_data(data)

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy over sampled data without PCA: ", accuracy)
        print()
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_undersampled(self):
        data = self.pre_process_undersample(4110, "negative")

        x_train, x_test, y_train, y_test = self.process_and_split_data(data)

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy under sampled data without PCA: ", accuracy)
        print()
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_oversampled_PCA(self):
        data = self.pre_process_oversample(4110, "positive")

        x_train, x_test, y_train, y_test = self.process_split_PCA(data)

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy over sampled data after PCA: ", accuracy)
        print()
        features = np.vstack((x_train, x_test))
        labels = np.vstack((y_train[:, None], y_test[:, None]))

        cross_val_acc = self.cross_validation(tree, features, labels)
       
        return cross_val_acc, y_test, pred

    def decision_tree_undersampled_PCA(self):
        data = self.pre_process_undersample(4110, "negative")

        x_train, x_test, y_train, y_test = self.process_split_PCA(data)

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        pred = tree.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, pred)
        print("Decision tree - Accuracy under sampled data after PCA: ", accuracy)
        print()
        features = np.vstack((x_train, x_test))
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

    def process_and_split_data(self, data):
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
        variance = self.percentage_of_variance(features,"Abalone", 5)  
        # print("Feature variances: ", variance)
        #this shows that only the length, diameter and height contribute a greater percentage
        #drop other features

        features = self.PCA(features, 5)

        variance = self.percentage_of_variance(features, "Abalone", 5) 

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

        # self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

        return (x_train, x_test, y_train, y_test)

    def process_split_PCA(self, data):
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
        variance = self.percentage_of_variance(features, "Abalone", 5)  
        # print("Feature variances: ", variance)
        #this shows that only the length, diameter and height contribute a greater percentage
        #drop other features

        features = self.PCA(features, 5)
        variance = self.percentage_of_variance(features, "Abalone", 5) 
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

        return (x_train, x_test, y_train, y_test)

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
        lg_over = abalone.logistic_regression_oversampled()
        lg_over_PCA = abalone.logistic_regression_oversampled_PCA()
        lg_under = abalone.logistic_regression_undersampled()
        lg_under_PCA = abalone.logistic_regression_undersampled_PCA()
        dc_over = abalone.decision_tree_oversampled()
        dc_over_PCA = abalone.decision_tree_oversampled_PCA()
        dc_under = abalone.decision_tree_undersampled()
        dc_under_PCA = abalone.decision_tree_undersampled_PCA()

        accuracies = [lg_over[0],lg_under[0], dc_over[0], dc_under[0]]
        accuracies_labels = ["lg_over","lg_under", "dtc_over", "dtc_under"]
        col = ["yellow", "m", "grey", "pink", "blue", "red", "black", "brown", "green", "cyan"]
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot(seq, accuracies[i], color=col[i], label=accuracies_labels[i])
        
        plt.ylabel("Accuracy (%)")
        plt.title("Model accuracies")
        plt.xticks(seq)
        plt.legend()
        plt.savefig("Model accuracies for undersampling and oversampling unsing Logistic Reg and Decision tree.jpeg")    
        plt.show()

        accuracies = [lg_under_PCA[0], lg_over_PCA[0], dc_over_PCA[0], dc_under_PCA[0]]
        accuracies_labels = ["lg_under_PCA", "lg_over_PCA", "dtc_over_PCA", "dtc_under_PCA"]
        col = ["yellow", "m", "grey", "pink", "blue", "red", "black", "brown", "green", "cyan"]
        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot(seq, accuracies[i], color=col[i], label=accuracies_labels[i])
        
        plt.ylabel("Accuracy (%)")
        plt.title("Model accuracies with PCA")
        plt.xticks(seq)
        plt.legend()
        plt.savefig("Model accuracies for undersampling and oversampling unsing Logistic Reg and Decision tree with PCA.jpeg")
        plt.show()

        true_labels = [lg_over[1],lg_under[1], dc_over[1], dc_under[1]]
        predictions = [lg_over[2],lg_under[2], dc_over[2], dc_under[2]]
        col1 = ["yellow", "m", "grey", "pink"]
        col2 = ["blue", "red", "black", "brown", "green", "cyan"]
        accuracies_labels = ["lg_over","lg_under", "dtc_over", "dtc_under"]

        encoder = LabelEncoder()

        for i in range(4):
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
        plt.savefig("ROC Curve multiple curves.jpeg")
        plt.show()

        return


abalone = ClassImbalance()
abalone.read_data("/Users/oyinlola/Desktop/MSc Data Science/SCC403 - Data Mining/Coursework/abalone19.csv")
# undersampled = abalone.pre_process_undersample(4110, "negative") 
# oversampled = abalone.pre_process_oversample(4110, "positive")

# abalone.plot_imbalance()


# abalone.logistic_regression_oversampled()
# abalone.logistic_regression_oversampled_PCA()
# abalone.logistic_regression_undersampled()
# abalone.logistic_regression_undersampled_PCA()
# abalone.decision_tree_oversampled()
# abalone.decision_tree_oversampled_PCA()
# abalone.decision_tree_undersampled()
# abalone.decision_tree_undersampled_PCA()

abalone.plot_metrics()