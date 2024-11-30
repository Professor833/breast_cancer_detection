from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


# Load the data
dataset = pd.read_csv('./dataset/breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# train the model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)

# MAking the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# computing the accuracy with k-Fold Cross Validation
'''
The cross_val_score function from the sklearn.model_selection module is used to
perform k-Fold Cross Validation. The estimator parameter of the cross_val_score
function is the classifier that we want to evaluate. The X parameter contains the
features, and the y parameter contains the target variable. The cv parameter
specifies the number of folds, which is 10 in this case. The cross_val_score function
returns the accuracies of all folds. The mean and standard deviation of the accuracies
are calculated and printed to the console.
'''
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
