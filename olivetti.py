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
from sklearn.ensemble import RandomForestClassifier
import warnings

def compare_KNN(models):
    model_metrics = []

    for name, model in models:

        clf=model

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)

        print("##############################")
        print("{} Accuracy score:{:0.2f}".format(name, metrics.accuracy_score(y_test, y_pred)))
        model_metrics.append(metrics.accuracy_score(y_test, y_pred))

        plt.figure(1, figsize=(8,8))
        sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
        plt.title("{} heat map without PCA".format(name).upper())
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()



    #visualize the performance of KNN models with different K value
    plt.figure(1, figsize=(8,8))
    plt.bar([name for name,model in models], model_metrics, align='center')
    plt.title("KNN performance for different K values")
    plt.xticks([name for name,model in models])
    plt.ylabel("Performance")
    for index, value in enumerate(model_metrics):
        plt.text(index, value, "%.2f" % value)
    plt.show()

    return model_metrics

def compare_models(models):
    # used for table printout
    model_metrics = []
    # used for bargraph to visualize performance
    model_scores = []

    for name, model in models:

        clf=model

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)

        print("##############################")
        print("{} Accuracy score:{:0.2f}".format(name, metrics.accuracy_score(y_test, y_pred)))
        model_scores.append(metrics.accuracy_score(y_test, y_pred))
        model_metrics.append([name, metrics.precision_score(y_test, y_pred, average='weighted'), metrics.recall_score(y_test, y_pred, average='weighted'), metrics.f1_score(y_test, y_pred, average='weighted'), mean_squared_error(y_test, y_pred)])

        plt.figure(1, figsize=(8,8))
        sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
        plt.title("{} heat map without PCA".format(name).upper())
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()

    #visualize the performance of the models
    plt.figure(1, figsize=(8,8))
    plt.bar([name for name,model in models], model_scores, align='center')
    plt.title("Model performance without PCA")
    plt.xticks([name for name,model in models])
    plt.ylabel("Performance")
    for index, value in enumerate(model_scores):
        plt.text(index, value, "%.2f" % value)
    plt.show()

    return model_metrics

#metrics printout
def model_metrics_printout(columns, metrics):
    print('\n{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format(*columns))
    for row in metrics:
        #print(row)
        print('{:<10}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>10.3f}'.format(*row))
    print()

def PCA_preprocess(n_components, X):
    pca=PCA(n_components=n_components, whiten=True)
    pca.fit(X)
    #print(pca.explained_variance_ratio_)
    print("Cumulative variance explained by {} components: {:.2%}".format(n_components, sum(pca.explained_variance_ratio_)))

    fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.imshow(pca.mean_.reshape((64,64)), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Average Face in dataset')
    plt.show()

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca

def PCA_compare_models(models):
    # used for table printout
    model_metrics = []
    # used for bargraph to visualize performance
    model_scores = []

    for name, model in models:

        clf=model

        clf.fit(X_train_pca, y_train)
        y_pred=clf.predict(X_test_pca)

        print("##############################")
        print("{} Accuracy score:{:0.2f}".format(name, metrics.accuracy_score(y_test, y_pred)))
        model_scores.append(metrics.accuracy_score(y_test, y_pred))
        model_metrics.append([name, metrics.precision_score(y_test, y_pred, average='weighted'), metrics.recall_score(y_test, y_pred, average='weighted'), metrics.f1_score(y_test, y_pred, average='weighted'), mean_squared_error(y_test, y_pred)])

        plt.figure(1, figsize=(8,8))
        sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
        plt.title("{} heat map with PCA".format(name).upper())
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()

    #visualize the performance of the models
    plt.figure(1, figsize=(8,8))
    plt.bar([name for name,model in models], model_scores, align='center')
    plt.title("Model performance with PCA")
    plt.xticks([name for name,model in models])
    plt.ylabel("Performance")
    for index, value in enumerate(model_scores):
        plt.text(index, value, "%.2f" % value)
    plt.show()

    return model_metrics

if __name__ == '__main__':
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

    #heatmap visualization
    corr = np.corrcoef(X_train)
    plt.figure(1, figsize=(8,8))
    sns.heatmap(corr)
    plt.show()

    warnings.filterwarnings('ignore')

    #KNN different K values
    print("\nTraining KNN for different K values")
    knn_models=[]
    knn_models.append(("1-NN",KNeighborsClassifier(n_neighbors=1)))
    knn_models.append(("2-NN",KNeighborsClassifier(n_neighbors=2)))
    knn_models.append(("3-NN",KNeighborsClassifier(n_neighbors=3)))
    knn_models.append(("4-NN",KNeighborsClassifier(n_neighbors=4)))
    knn_models.append(("5-NN",KNeighborsClassifier(n_neighbors=5)))
    knn_models.append(("6-NN",KNeighborsClassifier(n_neighbors=6)))
    knn_models.append(("7-NN",KNeighborsClassifier(n_neighbors=7)))

    knn_models_scores = compare_KNN(knn_models)

    max_knn_score = max(knn_models_scores)
    max_knn_score_index = knn_models_scores.index(max_knn_score)
    max_knn_score_k_value = knn_models[max_knn_score_index][0][0] #as a STRING


    #machine learning models
    print("\nTraining different machine learning models")
    models=[]
    #the bottom essentially does the following: models.append(("KNN w/K=1,KNeighborsClassifier(n_neighbors=1)))
    #but I did not want to hard code the 1 (which is the k with higherst score)
    models.append(("KNN w/K={}".format(max_knn_score_k_value),KNeighborsClassifier(n_neighbors=int(max_knn_score_k_value))))
    models.append(("DT",DecisionTreeClassifier()))
    models.append(("SVM",SVC()))
    models.append(("RF",RandomForestClassifier(n_estimators=64, random_state=0)))

    models_scores = compare_models(models)
    columns = ['Classifier', 'Precision', 'Recall', 'F1 Score', 'RMSE']
    model_metrics_printout(columns, models_scores)

    #PCA
    print("Training different machine learning models with PCA")
    X_train_pca, X_test_pca = PCA_preprocess(50, X)

    PCA_models_scores = PCA_compare_models(models)
    model_metrics_printout(columns, PCA_models_scores)

    print("X_train_pca shape:",X_train_pca.shape)
