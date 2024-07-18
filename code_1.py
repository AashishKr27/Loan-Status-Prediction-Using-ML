import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
loan_dataset = pd.read_csv('dataset.csv')
print(loan_dataset.head())
print(loan_dataset.shape)
print(loan_dataset.describe())
loan_dataset.isnull().sum()

# Drop the null values
loan_dataset = loan_dataset.dropna()

# Label encoding
loan_dataset.replace({"Loan_Status":{'N':0, 'Y':1}}, inplace=True)

# Dependent column values
loan_dataset['Dependents'].value_counts()
loan_dataset.replace({"Dependents":{'3+':4}}, inplace=True)
loan_dataset['Dependents'].value_counts()

# Data Visualization

# Education & Loan Status
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.show()

# Marital Status & Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.show()

# Convert categorical columns to numerical values
loan_dataset.replace({"Married":{'No':0, 'Yes':1}}, inplace=True)
loan_dataset.replace({"Education":{'Not Graduate':0, 'Graduate':1}}, inplace=True)
loan_dataset.replace({"Self_Employed":{'No':0, 'Yes':1}}, inplace=True)
loan_dataset.replace({"Gender":{'Female':0, 'Male':1}}, inplace=True)
loan_dataset.replace({"Property_Area":{'Rural':0, 'Semiurban':1, 'Urban':2}}, inplace=True)

# Split the data into X and y
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

print(X)
print(Y)

# Split the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Train the model
classifier = svm.SVC(kernel='linear')

# Fit the model
classifier.fit(X_train, Y_train)

# Predict the training set
train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(train_prediction, Y_train)
print('Accuracy on training data : ', train_accuracy)

# Predict the test set
test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(test_prediction, Y_test)
print('Accuracy on test data : ', test_accuracy)

# Predict the new data
input_data=(1,1,4,1,0,3036,2504,158,360,0,1)
input_data_as_numpy_array=np.asarray(input_data)
prediction=classifier.predict(input_data_as_numpy_array.reshape(1,-1))
print(prediction)

if prediction[0] == 1:
    print("Loan Status : Approved")
else:
    print("Loan Status : Rejected")