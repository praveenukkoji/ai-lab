# importing the required packages

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# load the dataset

iris = datasets.load_iris()
print('iris dataset loaded...')

# split data into the train and test samples

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)

print('dataset split into training and testing...')
print('Size of training data and its label', x_train.shape, y_train.shape)
print('Size of testing data and its label', x_test.shape, y_test.shape)

# print label number and their names

for i in range(len(iris.target_names)):
    print("label", i, " - ", str(iris.target_names[i]))

# create object of KNN Classifier

classifier = KNeighborsClassifier(n_neighbors=1)

# perform training

classifier.fit(x_train, y_train)

# perform testing

y_pred = classifier.predict(x_test)

# display results

print("Result of classification using KNN with k=1")

for r in range(0, len(x_test)):
    print("Sample: ", str(x_test[r]), "Actual-label: ", str(y_test[r]), "Predicted-label: ", str(y_pred))

print("Classification Accuracy: ", classifier.score(x_test, y_test))