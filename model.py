import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

# LOADING THE UNDELETED FEATURE SET DISTRIBUTION
csv_featset_normal = pd.read_csv(r"E:\project papers cite\NOTES\originaldistribution.csv",header=1,dtype='unicode')
csv_featset_normal= csv_featset_normal.sample(n=95564)
# Load the DELTED FEATURE SET DISTRIBUTION
csv_deleted_fname = pd.read_csv(r"E:\project papers cite\NOTES\testfeaturedistribution.csv", header=1,dtype='unicode')



################## MODEL EVALUATION FOR BODY FIELD METRICS #####################
def clf_a():
    print('MODEL EVALUATION FOR BODY FIELD METRICS')
    bodytest =csv_deleted_fname.iloc[: , :26]
    bodyoriginal =csv_featset_normal.iloc[: , :26]
    #LABELLING THE DATASET FOR DELETED AND UNDELETD FEATURES
    bodyoriginal['classlable']=0
    bodytest['classlable']=1
    print('Shape of the dataset')
    print(bodyoriginal.shape, bodytest.shape)
    #Arracge the stack arrays in sequence vertically(row wise) using vstack function
    clf_dataset = np.vstack((bodyoriginal, bodytest))
    #Creating the Validation set
    X = clf_dataset[:, :-1]
    y = clf_dataset[:, -1]
    y=y.astype('int')
    #Shape of the Validation
    print('Shape of Validation set')
    print (X.shape, y.shape)
    print("Evaluating Correa and Sureka Approach")
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),algorithm="SAMME.R", n_estimators=100, learning_rate=1.0)
    scoring_funcs = ['f1', 'accuracy', 'roc_auc']
    for scoring_func in scoring_funcs:
        print (scoring_func)
        scores = cross_val_score(model, X, y, cv=10, scoring=scoring_func)
        print(scores)
        print (np.around(100*np.mean(scores), 2), np.around(100*np.var(scores), 2))
    ''' We evaluate the model under different classifiers by Splitting the Test Data '''
    print('We evaluate the model under different classifiers using train_test_split method')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    models =[RandomForestClassifier(),KNeighborsClassifier(),LogisticRegression(solver='lbfgs', max_iter=15000)]
    for model1 in models:
      #Train the models
      model1.fit(X_train,y_train)
      test_prediction=model1.predict(X_test)
      accuracy=accuracy_score(y_test,test_prediction)
      f1=f1_score(y_test,test_prediction)
      roc_auc=roc_auc_score(y_test,test_prediction)
      recall= recall_score(y_test,test_prediction)
      precision=precision_score(y_test,test_prediction)

      print("Accuracy score of the model is ",model1,'=',accuracy)
      print("F1 score of the model is ",model1,'=',f1)
      print("ROC score of the model is ",model1,'=',roc_auc)
      print("Recall score of the model is ",model1,'=',recall)
      print("precision score of the model is ",model1,'=',precision)
      print("Done")
################## MODEL EVALUATION FOR BODY FIELD METRICS #####################
def clf_ab():

    print('MODEL EVALUATION FOR TITLE FIELD METRICS')
    titletest=csv_deleted_fname.iloc[:,31:57]
    titleoriginal =csv_featset_normal.iloc[:,31:57]
    #LAbeling the data for deleted and undeleted questions
    titletest['classlable']=1
    titleoriginal['classlable']=0
    print('Shape of the dataset')
    print(titleoriginal.shape,titletest.shape)
    #Arracge the stack arrays in sequence vertically(row wise) using vstack function
    clf_dataset = np.vstack((titleoriginal, titletest))
    #Creating the validationset
    X = clf_dataset[:, :-1]
    y = clf_dataset[:, -1]
    y=y.astype('int')
    print('Shape of Validation set')
    print (X.shape, y.shape)
    print("Evaluating Correa and Sureka Approach")
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),algorithm="SAMME.R", n_estimators=100, learning_rate=1.0)
    scoring_funcs = ['f1', 'accuracy', 'roc_auc']
    for scoring_func in scoring_funcs:
        print (scoring_func)
        scores = cross_val_score(model, X, y, cv=10, scoring=scoring_func)
        print(scores)
        print (np.around(100*np.mean(scores), 2), np.around(100*np.var(scores), 2))
    ''' We evaluate the model under different classifiers by Splitting the Test Data '''
    print('We evaluate the model under different classifiers using train_test_split method')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    models =[RandomForestClassifier(),KNeighborsClassifier(),LogisticRegression(solver='lbfgs', max_iter=15000)]
    for model1 in models:
      model1.fit(X_train,y_train)
      test_prediction=model1.predict(X_test)
      accuracy=accuracy_score(y_test,test_prediction)
      f1=f1_score(y_test,test_prediction)
      roc_auc=roc_auc_score(y_test,test_prediction)
      recall= recall_score(y_test,test_prediction)
      precision=precision_score(y_test,test_prediction)

      print("Accuracy score of the model is ",model1,'=',accuracy)
      print("F1 score of the model is ",model1,'=',f1)
      print("ROC score of the model is ",model1,'=',roc_auc)
      print("Recall score of the model is ",model1,'=',recall)
      print("precision score of the model is ",model1,'=',precision)
      print("Done")

################## MODEL EVALUATION FOR ALL(BODY< TITLE AND TAG) FIELD METRICS #####################
def clf_abc():
    print("Overall Approach")
    bodytest =csv_deleted_fname.iloc[: , :26]
    bodyoriginal =csv_featset_normal.iloc[: , :26]
    titletest=csv_deleted_fname.iloc[:, 31:57]
    titleoriginal =csv_featset_normal.iloc[:, 31:57]
    no_of_tags_test=csv_deleted_fname.iloc[:, 62]
    no_of_tags_original=csv_featset_normal.iloc[:, 62]
    #Feature Selection
    testfeatures = pd.concat([bodytest, titletest,no_of_tags_test], axis=1)
    originalfeatures= pd.concat([bodyoriginal, titleoriginal,no_of_tags_original], axis=1)
    #Classify the labels as deleted and undleted using binary classification
    originalfeatures['classlable']=0
    testfeatures['classlable']=1

    #originalfeatures=originalfeatures.sample(n=500)
    #testfeatures=testfeatures.sample(n=500)

    print("Shape of the model")
    print(originalfeatures.shape, testfeatures.shape)

    #Arracge the stack arrays in sequence vertically(row wise) using vstack function
    clf_dataset = np.vstack((originalfeatures, testfeatures))
    #Ceating the Validation set
    X = clf_dataset[:, :-1]
    y = clf_dataset[:, -1]
    y=y.astype('int')
    #Shape of Validation
    print (X.shape, y.shape)
    # Evaluating Correa and Sureka Approach with their model settings
    print("Evaluating Correa and Sureka Approach ")
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),algorithm="SAMME.R", n_estimators=100, learning_rate=1.0)
    scoring_funcs = ['f1', 'accuracy', 'roc_auc']
    y_pred = cross_val_predict(model, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_mat, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i,s=conf_mat[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    for scoring_func in scoring_funcs:
        print(scoring_func)
        scores = cross_val_score(model, X, y, cv=10, scoring=scoring_func)
        #Score of all random samples
        print(scores)
        #Mean of all the sample scores
        print (np.around(100*np.mean(scores), 2), np.around(100*np.var(scores), 2))
    '''# We evaluate the model under different classifiers by Splitting the Test Data '''
    print('We evaluate the model under different classifiers using train_test_split method')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    models =[RandomForestClassifier(),KNeighborsClassifier(),LogisticRegression(solver='lbfgs', max_iter=15000)]
    for model1 in models:
        #Training the models
        model1.fit(X_train,y_train)
        test_prediction=model1.predict(X_test)
        #Calculating the metrics value
        accuracy=accuracy_score(y_test,test_prediction)
        f1=f1_score(y_test,test_prediction)
        roc_auc=roc_auc_score(y_test,test_prediction)
        recall= recall_score(y_test,test_prediction)
        precision=precision_score(y_test,test_prediction)
        # CONFUSION MATRIX''''''
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=test_prediction)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()

        print("Accuracy score of the model is ",model1,'=',accuracy)
        print("F1 score of the model is ",model1,'=',f1)
        print("ROC score of the model is ",model1,'=',roc_auc)
        print("Recall score of the model is ",model1,'=',recall)
        print("precision score of the model is ",model1,'=',precision)
        print("Done")


print ('***** A ****')
#clf_a()

print ('***** A + B ****')
clf_ab()

print ('***** A + B + C ****')
#clf_abc()
