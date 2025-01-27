from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import train
from sklearn.metrics import classification_report, confusion_matrix


from pickle import dump, load
import pandas as pd
import numpy as np
from math import floor

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# import random state
np.random.seed(42)


# model definitions
class SklearnLogisticRegression:
    def __init__(self, epochs):
        self.model = linear_model.LogisticRegression(
            solver='liblinear', max_iter=epochs)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, X, Y):
        return self.model.score(X, Y)


class SVMClassifier:
    def __init__(self, kernel='linear', degree=3, gamma='scale', C=1.0):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.C = C
        self.model = SVC(kernel=self.kernel, degree=self.degree,
                         gamma=self.gamma, C=self.C, random_state=42)

    def fit(self, X, Y):
        """Train the SVM model."""
        self.model.fit(X, Y)

    def predict(self, X):
        """Predict using the trained SVM model."""
        return self.model.predict(X)

    def accuracy(self, X, Y):
        """Calculate and return the accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(Y, predictions)


class SklearnDecisionTree:
    def __init__(self):
        self.model = tree.DecisionTreeClassifier()

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, X, Y):
        return self.model.score(X, Y)


class SklearnKNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, X, Y):
        return self.model.score(X, Y)


# get data from our dataset file
X_train, Y_train, X_test, Y_test = train.get_data()


def find_hyper_params_log(model, X_train, Y_train):
    params = [
        {
            'penalty': ['l2', 'l1', 'elasticnet'],
            'C': [0.001, 0.01, 0.1, 1],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'max_iter': [100, 200, 300, 400, 500, 1000, 2000, 5000],
        }
    ]

    print("The GridSearchCV is running, this will take a while...")
    gridsearch = model_selection.GridSearchCV(
        estimator=model, param_grid=params, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
    grid_res = gridsearch.fit(X_train, Y_train)
    print("Done. \n")
    print("Best params:", gridsearch.best_params_)
    return gridsearch.best_estimator_


def find_hyper_params_tree(model, X_train, Y_train):
    params = [
        {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [1, 5, 10, 20],
            'min_samples_split': [2, 5, 10, 20],
        }
    ]
    print("The GridSearchCV is running, this will take a while...")
    gridsearch = model_selection.GridSearchCV(
        estimator=model, param_grid=params, scoring='accuracy', cv=3, verbose=0, n_jobs=-1)
    grid_res = gridsearch.fit(X_train, Y_train)
    print("Done. \n")
    print("Best params:", gridsearch.best_params_)
    return gridsearch.best_estimator_


def find_hyper_params_knn(model, X_train, Y_train):
    params = {
        'n_neighbors': [1, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500],
        'algorithm': ['auto'],
        'leaf_size': [20, 40],
        'weights': ['uniform'],
        'metric': ('minkowski', 'chebyshev')
    }
    print("The GridSearchCV is running, this will take a while...")
    gridsearch = model_selection.GridSearchCV(
        estimator=model, param_grid=params, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    gridsearch.fit(X_train, Y_train)
    print("Done. \n")
    print("Best params:", gridsearch.best_params_)
    return gridsearch.best_estimator_


# Calculate TPR (Sensitivity) and Objective Value
# We will use this to compare the models 
tpr_weight = 0.25

def calculate_metrics(model, X_test, Y_test):
    predicts = model.predict(X_test)
    cm = confusion_matrix(Y_test, predicts)
    accuracy = accuracy_score(Y_test, predicts)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    objective_value = accuracy + sensitivity * tpr_weight
    return accuracy, sensitivity, objective_value



#epochs = 400
#sklearn_log = SklearnLogisticRegression(epochs=epochs)
#sklearn_log.fit(X_train,Y_train)
#print("Accuracy of Sklearn logistic reg (no hyperparameter tuning)", sklearn_log.accuracy(X_test,Y_test))
#Accuracy of Sklearn logistic reg (no hyperparameter tuning) 0.7277730408495465

#best_log = find_hyper_params_log(sklearn_log.model, X_train, Y_train)
#print("Accuracy of Sklearn logistic reg (with hyperparameter tuning) (no scaling)", best_log.score(X_test, Y_test))
#Accuracy of Sklearn logistic reg (with hyperparameter tuning) (no scaling) 0.727540500736377

#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

#best_log = find_hyper_params_log(sklearn_log.model, X_train, Y_train)
#print("Accuracy of Sklearn logistic reg (with hyperparameter tuning) (with scaling)", best_log.score(X_test, Y_test))
#Accuracy of Sklearn logistic reg (with hyperparameter tuning) (with scaling) 0.726610340283699


#with open('log_reg.pkl', 'rb') as f:
#    best_log = load(f)

#best_log.fit(X_train, Y_train)
#print("Accuracy of Sklearn logistic reg (with hyperparameter tuning) (with scaling)", best_log.score(X_test, Y_test))


#sklearn_tree = SklearnDecisionTree()
#sklearn_tree.fit(X_train, Y_train)
#print("Accuracy of sklearn decision tree (no hyperparameter tuning)", sklearn_tree.accuracy(X_test,Y_test))

#best_tree = find_hyper_params_tree(sklearn_tree.model, X_train, Y_train)
#print("Accuracy of sklearn decision tree (with hyperparameter tuning)", best_tree.score(X_test, Y_test))

# As mentioned in our Milestone 2 that these values were nearly the same.
# Output:
#Accuracy of Sklearn logistic reg (no hyperparameter tuning) 0.7277730408495465
#Accuracy of Sklearn logistic reg (with hyperparameter tuning) (no scaling) 0.727540500736377
#Accuracy of Sklearn logistic reg (with hyperparameter tuning) (with scaling) 0.726610340283699



# Logistic Regression
print("#" * 50)
print("\nLogistic Regression")
with open('log_reg.pkl', 'rb') as f:
    best_log = load(f)
predicts = best_log.predict(X_test)
print("Confusion Matrix \n", confusion_matrix(Y_test, predicts))
print("Classification Report \n", classification_report(Y_test, predicts))
print("ROC AUC Score \n", roc_auc_score(Y_test, best_log.predict(X_test)))
log_accuracy, log_sensitivity, log_objective = calculate_metrics(best_log, X_test, Y_test)
print(f"Accuracy: {log_accuracy * 100:.2f}%")
print(f"Sensitivity: {log_sensitivity:.2f}")
print(f"Objective Value: {log_objective:.2f}")

# Decision Tree
print("#" * 50)
print("\nDecision Tree")
with open('decision_tree.pkl', 'rb') as f:
    best_tree = load(f)
predicts = best_tree.predict(X_test)
print("Confusion Matrix \n", confusion_matrix(Y_test, predicts))
print("Classification Report \n", classification_report(Y_test, predicts))
print("ROC AUC Score \n", roc_auc_score(Y_test, best_tree.predict(X_test)))
tree_accuracy, tree_sensitivity, tree_objective = calculate_metrics(best_tree, X_test, Y_test)
print(f"Accuracy: {tree_accuracy * 100:.2f}%")
print(f"Sensitivity: {tree_sensitivity:.2f}")
print(f"Objective Value: {tree_objective:.2f}")

# SVM
print("#" * 50)
print("\nSVM")
if os.path.exists('svm.pkl'):
    with open('svm.pkl', 'rb') as f:
        best_svm = load(f)
    print("Loaded SVM model from svm.pkl")
else:
    svm_rbf = SVMClassifier(kernel='rbf', gamma=0.01, C=1.0)
    svm_rbf.fit(X_train, Y_train)
    best_svm = svm_rbf
    with open('svm.pkl', 'wb') as f:
        dump(best_svm, f, protocol=5)
    print("Saved SVM model to svm.pkl")

# To save time, we will only tune the RBF kernel SVM, as it is the best performing model in testing. SVM took 10 mins on average on each run/iteration.

svm_predictions = best_svm.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(Y_test, svm_predictions))
print("Classification Report:\n", classification_report(Y_test, svm_predictions))
print("ROC AUC Score:", roc_auc_score(Y_test, svm_predictions))
svm_accuracy, svm_sensitivity, svm_objective = calculate_metrics(best_svm, X_test, Y_test)
print(f"Accuracy: {svm_accuracy * 100:.2f}%")
print(f"Sensitivity: {svm_sensitivity:.2f}")
print(f"Objective Value: {svm_objective:.2f}")

# KNN
print("#" * 50)
print("\nKNN")

if os.path.exists('knn.pkl'):
    with open('knn.pkl', 'rb') as f:
        best_knn = load(f)
    print("Loaded KNN model from knn.pkl")
else:
    knn_model = SklearnKNN(n_neighbors=5)
    knn_model.fit(X_train, Y_train)
    knn_accuracy = knn_model.accuracy(X_test, Y_test)
    print(
        f"Accuracy of KNN (no hyperparameter tuning): {knn_accuracy * 100:.2f}%")
    knn_model_h = SklearnKNN()
    best_knn = find_hyper_params_knn(knn_model_h.model, X_train, Y_train)
    knn_tuned_accuracy = best_knn.score(X_test, Y_test)
    print(
        f"Accuracy of KNN (with hyperparameter tuning): {knn_tuned_accuracy * 100:.2f}%")
    with open('knn.pkl', 'wb') as f:
        dump(best_knn, f, protocol=5)
    print("Saved KNN model to knn.pkl")
knn_predictions = best_knn.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(Y_test, knn_predictions))
print("Classification Report:\n", classification_report(Y_test, knn_predictions))
print("ROC AUC Score:", roc_auc_score(Y_test, knn_predictions))
knn_accuracy, knn_sensitivity, knn_objective = calculate_metrics(best_knn, X_test, Y_test)
print(f"Accuracy: {knn_accuracy * 100:.2f}%")
print(f"Sensitivity: {knn_sensitivity:.2f}")
print(f"Objective Value: {knn_objective:.2f}")


# Collect results into a DataFrame for plotting and comparison
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Support Vector Machine'],
    'Accuracy': [log_accuracy, tree_accuracy, knn_accuracy, svm_accuracy],
    'Sensitivity': [log_sensitivity, tree_sensitivity, knn_sensitivity, svm_sensitivity],
    'Objective Value': [log_objective, tree_objective, knn_objective, svm_objective]
})

print("\nComparison Table:")
print(models)

# Plot results
import matplotlib.pyplot as plt
import seaborn as sns

# Plot accuracy and sensitivity
df_plotting = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Support Vector Machine',
              'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Support Vector Machine'],
    'Values': [log_accuracy, tree_accuracy, knn_accuracy, svm_accuracy,
               log_sensitivity, tree_sensitivity, knn_sensitivity, svm_sensitivity],
    'Type': ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy',
             'Sensitivity', 'Sensitivity', 'Sensitivity', 'Sensitivity']
})

plt.figure(figsize=(15, 10))
sns.barplot(y=df_plotting['Values'], x=df_plotting['Type'], hue=df_plotting['Model'], orient="v")
plt.ylim(min(df_plotting['Values']) * 0.99, max(df_plotting['Values']) * 1.01)
plt.title("Model Comparison: Accuracy and Sensitivity")
plt.show()

# Plot objective value
plt.figure(figsize=(10, 5))
sns.barplot(y=models['Objective Value'], x=models['Model'], orient="v")
plt.ylim(min(models['Objective Value']) * 0.9, max(models['Objective Value']) * 1.01)
plt.title("Objective Value Comparison")
plt.show()
