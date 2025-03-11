import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def encoding(self, target):
        self.data[target].replace('Iris-setosa', 0, inplace = True)
        self.data[target].replace('Iris-versicolor', 1, inplace = True)
        self.data[target].replace('Iris-virginica', 2, inplace = True)
    
    def split_data(self, target):
        self.input_df = self.data.drop([target, 'Id'], axis=True)
        self.output_df = self.data[target]
    
class TrainTestSplitData:
    def __init__(self, input_df, output_df):
        self.input_df = input_df
        self.output_df = output_df
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    
    def split(self, test_size = 0.2, random_state = 42):
        self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(self.input_df, self.output_df, test_size= test_size, random_state=random_state)
        

class ModelHandler:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.createModel()
        
   
    def createModel(self):
        self.model = XGBClassifier(random_state = 42)
    
    def trainModel(self):
        self.model.fit(self.x_train, self.y_train)
        
    def makePredictions(self):
        self.y_pred = self.model.predict(self.x_test)
        
    def evaluateModel(self):
        return accuracy_score(self.y_test, self.y_pred)
    
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_pred, target_names=['0','1','2']))
        
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  # Open the file in write-binary mode
            pickle.dump(self.model, file)

file_path = 'Iris.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.encoding('Species')
data_handler.split_data('Species')
input_df = data_handler.input_df
output_df = data_handler.output_df

traintest_split = TrainTestSplitData(input_df, output_df)
traintest_split.split()
x_train = traintest_split.x_train
x_test = traintest_split.x_test
y_train = traintest_split.y_train
y_test = traintest_split.y_test

model_handler = ModelHandler(x_train, x_test, y_train, y_test)
model_handler.trainModel()
model_handler.makePredictions()
print("Model Accuracy:", model_handler.evaluateModel())
model_handler.createReport()

model_handler.save_model_to_file('xgb_model.pkl')    
        