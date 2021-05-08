import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error


#load data
data=np.load("./input/olivetti_faces.npy")
target=np.load("./input/olivetti_faces_target.npy") #face id

#check the shape of the data
print("Data shape {}".format(data.shape))
print("Target shape {}".format(target.shape))

X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
print("\nX reshaped:",X.shape)

X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
print("\nX_train shape:",X_train.shape)
print("y_train shape:",y_train.shape)

#print(X_train)

#Visualization
corr = np.corrcoef(X_train)
plt.figure(1, figsize=(8,8))
sns.heatmap(corr)
plt.show()

#PCA
pca = PCA()
pca.fit(X)

plt.figure(1, figsize=(8,8))

plt.plot(pca.explained_variance_, linewidth=2)

plt.title('PCA')
plt.xlabel('Components')
plt.ylabel('Explained Variances')
plt.show()

#below is hat gets added to the model_metrics array
#model_metrics = ['Classifier', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'RMSE']
#model_metrics = ['Classifier', 'Precision', 'Recall', 'F1 Score', 'RMSE']
model_metrics = []

#KNN different K values
print("\nTraining KNN for different K values")
knn_models=[]
knn_models_scores = []
knn_models.append(("1-NN",KNeighborsClassifier(n_neighbors=1)))
knn_models.append(("3-NN",KNeighborsClassifier(n_neighbors=3)))
knn_models.append(("5-NN",KNeighborsClassifier(n_neighbors=5)))

for name, model in knn_models:

    clf=model

    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    print("##############################")
    print("{} Accuracy score:{:0.2f}".format(name, metrics.accuracy_score(y_test, y_pred)))
    knn_models_scores.append(metrics.accuracy_score(y_test, y_pred))
    plt.figure(1, figsize=(8,8))
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
    plt.title("{} heat map without PCA".format(name).upper())
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


#visualize the performance of KNN models with different K value
plt.figure(1, figsize=(8,8))
plt.bar([name for name,model in knn_models], knn_models_scores, align='center')
plt.title("KNN performance for different K values")
plt.xticks([name for name,model in knn_models])
plt.ylabel("Performance")
for index, value in enumerate(knn_models_scores):
    plt.text(index, value, "%.2f" % value)
plt.show()

max_knn_score = max(knn_models_scores)
max_knn_score_index = knn_models_scores.index(max_knn_score)
max_knn_score_k_value = knn_models[max_knn_score_index][0][0] #as a STRING

#machine learning models
print("\nTraining different machine learning models")
models=[]
models_scores = []
#the bottom essentially does the following: models.append(("KNN w/K=1,KNeighborsClassifier(n_neighbors=1)))
#but I did not want to hard code the 1 (which is the k with higherst score)
models.append(("KNN w/K={}".format(max_knn_score_k_value),KNeighborsClassifier(n_neighbors=int(max_knn_score_k_value))))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))

for name, model in models:

    clf=model

    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    print("##############################")
    print("{} Accuracy score:{:0.2f}".format(name, metrics.accuracy_score(y_test, y_pred)))
    models_scores.append(metrics.accuracy_score(y_test, y_pred))
    #model_metrics.append([name, metrics.precision_score(y_test, y_pred, average='weighted'), metrics.recall_score(y_test, y_pred, average='weighted'), metrics.f1_score(y_test, y_pred, average='weighted'), metrics.roc_auc_score(y_test, y_pred, multi_class='ovr'), mean_squared_error(y_test, y_pred)])
    model_metrics.append([name, metrics.precision_score(y_test, y_pred, average='weighted'), metrics.recall_score(y_test, y_pred, average='weighted'), metrics.f1_score(y_test, y_pred, average='weighted'), mean_squared_error(y_test, y_pred)])
    plt.figure(1, figsize=(8,8))
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
    plt.title("{} heat map without PCA".format(name).upper())
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

plt.figure(1, figsize=(8,8))
plt.bar([name for name,model in models], models_scores, align='center')
plt.title("Models performance")
plt.xticks([name for name,model in models])
plt.ylabel("Performance")
for index, value in enumerate(models_scores):
    plt.text(index, value, "%.2f" % value)
#plt.show()

#metrics printout
#columns = ['Classifier', 'Precision', 'Recall', 'F1 Score', 'AUROC, ''RMSE']
columns = ['Classifier', 'Precision', 'Recall', 'F1 Score', 'RMSE']
print('\n{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(*columns))
for row in model_metrics:
    #print(row)
    print('{:<10}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}'.format(*row))
