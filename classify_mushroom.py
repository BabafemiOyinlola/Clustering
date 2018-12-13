import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

class  ClassifyMushroom:
    def __init__ (self):
        self.features = []
        self.labels = []

    def read_mushroom_data(self, filepath):
        read = open(filepath, "r")
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
            # data_array = []
            
            self.features = features
            self.labels = labels
            self.data = data_array

            return (data_array)
        
    #random forest with missing data predicted
    def random_forest_classifier1(self):
        '''Using random forests'''
        #predict missing data
        self.predict_missing_data()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        # reduce dimentionality
        name = "-Random forests. Missing value predicted"
        self.PCA(features_encoded)
        self.percentage_of_variance(features_encoded, name)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Random forests|Missing value predicted|CV")
        self.cross_validation(classifier, features_encoded, labels_encoded)
        self.metrics(pred, y_test, name)
        return

    #random forest with missing data filled with mode
    def random_forest_classifier2(self):
        '''Using random forests'''
        #Fill missing data with mode
        self.fill_missing_data_with_mode()

        #reduce dimentionality
        # self.features = self.PCA()
        # self.percentage_of_variance()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded =  hot_encoder.fit_transform(features_encoded)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)

        print("Random forests|Missing value filled with mode|CV")
        self.cross_validation(classifier, features_encoded, labels_encoded)
        self.metrics(pred, y_test)
        return

    #logistic regression with missing data predicted
    def logistic_regression1(self):
        '''Using Logistic Regression'''
        self.predict_missing_data()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        # reduce dimentionality
        name = "-Logistic Regression. Missing value predicted"
        # self.PCA(features_encoded)
        # self.percentage_of_variance(features_encoded, name)

        features_encoded = features_encoded.astype(np.float)
        labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value predicted|CV")
        self.cross_validation(reg, features_encoded, labels_encoded)
        print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test, name)
        return
   
    #logistic regression with missing data filled with mode
    def logistic_regression2(self):
        '''Using Logistic Regression'''
        self.fill_missing_data_with_mode()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        features_encoded = features_encoded.astype(np.float)
        labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
         
        print("Logistic Regression|Missing value filled with mode|CV")
        self.cross_validation(reg, features_encoded, labels_encoded)

        print("Logistics Regression 2 metrics: ")
        self.metrics(pred, y_test)
        return

    #logistic regression with missing data predicted
    def logistic_regression1_PCA(self):
        '''Using Logistic Regression'''
        self.predict_missing_data()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        hot_encoder.fit_transform(features_encoded)

        # reduce dimentionality
        self.PCA(features_encoded)
        self.percentage_of_variance(features_encoded, "- Logistic Regression. Missing value filled by prediction")


        features_encoded = features_encoded.astype(np.float)
        labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        self.cross_validation(reg, features_encoded, labels_encoded)
        print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test)
        return

    #logistic regression with missing data rows deleted
    def logistic_regression_del_rows(self):
        '''Using Logistic Regression'''
        self.del_missing_value_rows()

        name = "-Logistic Regression. Missing value rows deleted"

        features_encoded = np.array([])

        for col in range(self.features.shape[1]):
            temp = pd.get_dummies(self.features[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value feature deleted|CV")
        self.cross_validation(reg, features_encoded, self.labels)
        print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test, "e", name)
        return

    #logistic regression with missing data feature deleted
    def logistic_regression_del_feat(self):
        '''Using Logistic Regression'''
        self.delete_missing_val_feature()

        # reduce dimentionality
        name = "-Logistic Regression. Missing featured deleted"
        # self.PCA(features_encoded)
        # self.percentage_of_variance(features_encoded, name)

        features_encoded = np.array([])

        for col in range(self.features.shape[1]):
            temp = pd.get_dummies(self.features[:, col])
            temp = np.array(temp)
            if col == 0:
                features_encoded = temp
            else:
                features_encoded = np.column_stack([features_encoded, temp])

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, self.labels)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        print("Logistic Regression|Missing value feature deleted|CV")
        self.cross_validation(reg, features_encoded, self.labels)
        print("Logistics Regression 1 metrics: ")
        self.metrics(pred, y_test, "e", name)
        return

    #logistic regression with missing data filled with mode
    def logistic_regression2_PCA(self):
        '''Using Logistic Regression'''
        self.fill_missing_data_with_mode()

        labels_encoded = self.labels.copy()
        encode_l = preprocessing.LabelEncoder()
        labels_encoded = encode_l.fit_transform(self.labels)

        features_encoded = self.features.copy()
        encoder = preprocessing.LabelEncoder()
        for col in range(self.features.shape[1]):
            features_encoded[:, col] = encoder.fit_transform(self.features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        features_encoded = hot_encoder.fit_transform(features_encoded)

        # reduce dimentionality
        name = "- Logistic Regression. Missing value filled with mode"
        self.PCA(features_encoded)
        self.percentage_of_variance(features_encoded, name)

        features_encoded = features_encoded.astype(np.float)
        labels_encoded = labels_encoded.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(features_encoded, labels_encoded)

        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
 
        self.cross_validation(reg, features_encoded, labels_encoded)
        print("Logistics Regression 2 metrics: ")
        self.metrics(pred, y_test, name)
        return

    #handle missing data by predicting values
    def predict_missing_data(self):
        '''In the mushroom dataset, there are missing values. Here, it is handled by predicting with
        random forests'''
        #Select row with missing values and divide into test and training data used to train. 
        missing_value_col = self.features[:, 10]
        content = []
        val = []
        for item in range(len(missing_value_col)):
            if missing_value_col[item] == "?":
                val.append(missing_value_col[item])
            else:
                content.append(missing_value_col[item])
   
        content = np.array(content)
        val = np.array(val)

        new_features = np.delete(self.features, obj=10, axis=1) #drop col with missing values: '?'

        #encode features 
        new_features_encoded = new_features.copy()

        encoder = preprocessing.LabelEncoder()
        for col in range(new_features.shape[1]):
            new_features_encoded[:, col] = encoder.fit_transform(new_features[:, col])

        hot_encoder = preprocessing.OneHotEncoder()
        hot_encoder.fit_transform(new_features_encoded)

        random = np.arange(new_features_encoded.shape[0])
        np.random.shuffle(random)
        feature_train = new_features_encoded[random[:len(content)], :]
        feature_test = new_features_encoded[random[len(content):], :]

        feature_train = feature_train.astype(np.float)
        feature_test = feature_test.astype(np.float)

        #predict missing values using random forests
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(feature_train, content)
        missing_values = classifier.predict(feature_test)

        update_features = self.features
        count = 0

        #Fill features with predicted missing values
        for col in range(self.features.shape[0]):
            if self.features[col, 10] == "?":
                # print("Before: ",self.features[col,:])
                update_features[col, 10] = missing_values[count]
                count += 1
                # print("After: ",update_features[col,:])

        # print("Count end: ", count)
        if count == len(missing_values):
            self.features = update_features
            print("Missing values handled.")
 
        return

    #handle missing data by filling with mode
    def fill_missing_data_with_mode(self):
        miss_col = self.features[:, 10]
        
        feature_dict = {}

        for i in range(miss_col.shape[0]):
            if miss_col[i] in feature_dict:
                feature_dict[miss_col[i]] += 1
            else:
                feature_dict[miss_col[i]] = 1
        
        mode_num = 0
        mode = ""

        for key, value in feature_dict.items():
            if key != "?":
                if value > mode_num:
                    mode_num = value
                    mode = key
        
        #Impute missing values with mode
        count = 0
        for i in range(self.features.shape[0]):
            if self.features[i, 10] == "?":
                self.features[i, 10] = mode
                count += 1

        if mode_num >= count:
            return 
        else:
           print("Missing data not handled")
           return False

    #handle missing data by deleting feature contsining missing values (PCA argument)
    def delete_missing_val_feature(self):
        features = self.features
        new_feautures = np.delete(features, obj=10, axis=1)
        self.features = new_feautures   
        return
    
    def del_missing_value_rows(self):
        data = self.data.copy()
        length = data.shape[0]
        for i in range(length - 1, 0, -1):
            if data[i, 11] == "?":
                data = np.delete(data, obj=i, axis=0)
                length = data.shape[0]
                continue

        labels = data[:, 0]
        features =  np.delete(data, obj=0, axis=1)

        self.features = features
        self.labels = labels

        return

    def metrics(self, predictions, true_labels, recall_label, name=""):
        accuracy = metrics.accuracy_score(true_labels, predictions)
        confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
        recall = metrics.recall_score(true_labels, predictions, pos_label=recall_label)
        # precision = metrics.pr
        print("Accuracy: ",accuracy)
        print("Confusion matrix: \n", confusion_matrix)
        print("Recall: ",recall)
        # self.roc_curve_acc(true_labels, predictions, name, recall_label)
        return

    def PCA(self, data):
        pca = TruncatedSVD(n_components=22)
        pca.fit(data)
        cof = pca.components_
        trasform_data = pca.transform(data)
        return trasform_data

    def percentage_of_variance(self, data, name):
        pca = TruncatedSVD(n_components=22) #we have two components
        pca.fit(data)
        colums = ["cap-shape","cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", 
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", 
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
        plt.bar(colums, pca.explained_variance_ratio_, tick_label= colums)
        plt.xlabel("Principal Component")
        plt.ylabel("% Variance Explained")
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.title("Percentage of Variance")   
        plt.savefig("Percentage of variance " + name + ".jpeg")
        plt.show()
        return

    def cross_validation(self,algorithm, features, labels):
        accuracy = cross_val_score(algorithm, features, labels, scoring='accuracy', cv = 10)
        accuracy_percent = accuracy.mean() * 100
        print("Cross validation accuracy: " , accuracy_percent)

        return

    def roc_curve_acc(self, Y_test, Y_pred, name, pos_lab):
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=pos_lab)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        plt.title('ROC Curve')
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange',label='AUC = %0.3f'%(roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'b--')
        plt.ylim([-0.1, 1.1])
        plt.xlim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.tight_layout()
        plt.title("ROC Curve")   
        plt.savefig("ROC Curve " + name + ".jpeg")
        plt.show()
        return
