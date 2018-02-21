#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:28:16 2018

@author: madhuseelam
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as sm
import pprint
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def featureScaling(x):
    sc_X = StandardScaler()
    if isinstance(x, pd.core.frame.DataFrame):
        print ("in if")
        cols = x.columns
        return pd.DataFrame(data=sc_X.fit_transform(x), columns=cols)
    else:
        print("in else")
        return sc_X.fit_transform(x)


def getDataFrame(x_as_array, columns_as_array):
    return pd.DataFrame(data=x_as_array, columns=columns_as_array)

def onehotEncoder(xtrain, xtest, int_col_array):
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder(categorical_features=int_col_array)
    xtrain1 = onehotencoder.fit_transform(xtrain).toarray()
    xtest1 = onehotencoder.fit_transform(xtest).toarray()
    cols = xtrain.columns
    int_col_array = sorted (int_col_array , reverse=True)
    print(cols)
    print(int_col_array)
    new_cols = cols
    for i in int_col_array:
        a = len(xtrain.iloc[:, i].value_counts(dropna=False))
        print ("value count for column " + str(i) + cols[i] + ":" + str(a))
        for x in range(a-1):
            print(x, cols[i] + str(x))
            new_cols = new_cols.insert(i, cols[i] + str(x))

    print(new_cols)
    xtrain1 = pd.DataFrame(data=xtrain1, columns=new_cols)
    xtest1 = pd.DataFrame(data=xtest1, columns=new_cols)
    return xtrain1, xtest1

def getPCA(xtrain, xtest):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=None)
    pca.fit_transform(xtrain)
    pca.transform(xtest)
    explaine_variance = pca.explained_variance_ratio_
    for x in explaine_variance:
        print ("%0.5f" % x)

    array = [ x for x in explaine_variance  if (x > 0.1)]

    pca = PCA(n_components=len(array))
    x_train = pca.fit_transform(xtrain)
    x_test = pca.transform(xtest)
    return x_train, x_test


def logisticClassifer (x_train, y_train, x_test):
    #Logistic regression
    from sklearn.linear_model import LogisticRegression
    logis_classifier = LogisticRegression()
    logis_classifier.fit(x_train, y_train)
    return logis_classifier.predict(x_test)


def knnClassifer(x_train, y_train, x_test):
    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn_classifer = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn_classifer.fit(x_train, y_train)
    return knn_classifer.predict(x_test)


def svmClassifer(x_train, y_train, x_test):
     #SVM Support vector model
    from sklearn.svm import SVC
    svm_classifer = SVC(kernel="linear") #poly
    svm_classifer.fit(x_train, y_train)
    return svm_classifer.predict(x_test)

def ksvmClassifer(x_train, y_train, x_test):
    #Kernal SVM Support vector model
    from sklearn.svm import SVC
    ksvm_classifer = SVC(kernel="rbf") #poly
    ksvm_classifer.fit(x_train, y_train)
    return ksvm_classifer.predict(x_test)

def navieBayes(x_train, y_train, x_test):
    #Navie Bayes
    from sklearn.naive_bayes import GaussianNB
    nb_classifer = GaussianNB() #poly
    nb_classifer.fit(x_train, y_train)
    return nb_classifer.predict(x_test)


def DTClassifer(x_train, y_train, x_test):
    #Decission Tree
    from sklearn.tree import DecisionTreeClassifier
    dt_classifer = DecisionTreeClassifier(criterion="entropy") #poly
    dt_classifer.fit(x_train, y_train)
    return dt_classifer.predict(x_test)


def RFClassifer(x_train, y_train, x_test):
    #Random forest Decission Tree
    from sklearn.ensemble import RandomForestClassifier
    rf_classifer = RandomForestClassifier(n_estimators=1000) #, criterion="entropy" ) #poly
    rf_classifer.fit(x_train, y_train)
    return rf_classifer.predict(x_test)


def runAllCategoricalModels(x_train, y_train, x_test):
     logis_pred = logisticClassifer (x_train, y_train, x_test)
     knn_pred = knnClassifer(x_train, y_train, x_test)
     svm_pred = svmClassifer(x_train, y_train, x_test)
     ksvm_pred = ksvmClassifer(x_train, y_train, x_test)
     nb_pred = navieBayes(x_train, y_train, x_test)
     dt_pred = DTClassifer(x_train, y_train, x_test)
     rf_pred = RFClassifer(x_train, y_train, x_test)
     return logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred

def writeCsvFiles(pred, name, predColumnName, dataframe_col):
    new_y = pd.DataFrame(pred, columns=[predColumnName])
    new_y = new_y.replace(1, 'Y')
    new_y = new_y.replace(0, 'N')
    finaldata = pd.concat ([dataframe_col, new_y], axis=1)
    finaldata.to_csv('/bigdata/dat/'+name+'.csv', index = False)

def backwordElimination(x, y):
    ###Building optimal model using backward elimination
    ###R^2 represetns how good is your dataset.
    ###Closer to 1 the best
    ###The more variables (columns) the bigger the value. Add more columns witl increase the r^2
    ###Adjusted R^2 will reduce when we add more columns
    ###When R^2 and AR^2 reaches closest point then that is optimal model.
    ###Whenever Adjusted R^2 reduces stop eliminating
    y = list(y)
    x = pd.DataFrame(np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1),
                      columns=x.columns.insert(0, "dummy"))

    printValueCount(x)
    print(x.columns)
    print(len(y))
    print(len(x))

    # print (x2.shape)
    regress_ols = sm.OLS(endog=y, exog=x).fit()
    indexes = regress_ols.pvalues.sort_values(ascending=False)
    rsquared = regress_ols.rsquared
    arsquared = regress_ols.rsquared_adj
    print("rsquared %.5f arsquared %.5f" % (rsquared, arsquared))

    while (len(indexes) > 1):
        # print (("columns :%s values: %s" % (indexes.index, indexes.values))
        indexes = indexes[1:len(indexes)]
        regress_ols = sm.OLS(endog=y, exog=x.loc[:, indexes.index]).fit()
        indexes = regress_ols.pvalues.sort_values(ascending=False)

        if (arsquared < regress_ols.rsquared_adj):
            arsquared = regress_ols.rsquared_adj
        else:
            print("rsquared %.5f arsquared %.5f" % (regress_ols.rsquared, regress_ols.rsquared_adj))
            #print(tabulate(indexes))
            #print("indexes:", indexes)
            indexes = indexes.drop('dummy')
            for v in indexes.index:
                print('%s\t:\t%.7f' %(v , indexes[v]))
            return indexes

        rsquared = regress_ols.rsquared
        print("rsquared %.5f arsquared %.5f" % (rsquared, arsquared))
    return indexes

def preprocess(dataset, mf_imputer, labelencoder_X) :
    #dataset1.drop('Loan_ID', axis =1, inplace=True)
    #missingno.bar(dataset,color= sns.color_palette('viridis'), log=True)
    dataset['CoapplicantIncome'].fillna(0, inplace=True)
    dataset['Gender'].fillna('Male', inplace=True)
    dataset['Married'].fillna('Yes', inplace=True)
    dataset['Dependents'] = dataset['Dependents'].replace('3+', 4)
    dataset['Dependents'].fillna(0, inplace=True)
    dataset['Dependents'] = dataset['Dependents'].astype(int)



    dataset['Self_Employed'].fillna('No', inplace=True)
    #df[‘Credit_History’].fillna(df[‘Loan_Status’], inplace = True)

    #remove rows if the LoanAmount is null. There is not point processing whrn loan is null.
    dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(), inplace=True)
    #dataset = dataset[dataset['LoanAmount'].notnull()]
    dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean(), inplace=True)

    dataset['Credit_History'] = mf_imputer.fit_transform(dataset[['Credit_History']]).ravel()

    # Encoding categorical data
    # Encoding the Independent Variable

    dataset['Gender'] = labelencoder_X.fit_transform(dataset['Gender'])
    dataset['Married'] = labelencoder_X.fit_transform(dataset['Married'])
    dataset['Education'] = labelencoder_X.fit_transform(dataset['Education'])
    dataset['Self_Employed'] = labelencoder_X.fit_transform(dataset['Self_Employed'])
    dataset['Property_Area'] = labelencoder_X.fit_transform(dataset['Property_Area'])
    dataset['Credit_History'] = labelencoder_X.fit_transform(dataset['Credit_History'])
    #dataset['Loan_Status'] = labelencoder_X.fit_transform(dataset['Loan_Status'])


    return dataset


def printAccuracy_score(y_test, logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred):
    print( "logistic accuracy_score" , accuracy_score(y_test, logis_pred))
    print( "KNN accuracy_score" , accuracy_score(y_test, knn_pred))
    print( "SVM accuracy_score" , accuracy_score(y_test, svm_pred))
    print( "KSVM -rbf accuracy_score" , accuracy_score(y_test, ksvm_pred))
    print( "Navie Bayes accuracy_score" , accuracy_score(y_test, nb_pred))
    print( "DT accuracy_score" , accuracy_score(y_test, dt_pred))
    print( "RF accuracy_score" , accuracy_score(y_test, rf_pred))

def printConfusion_matrix(y_test, logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred):
    print ("logistic CM\n" ,confusion_matrix(y_test, logis_pred))
    print ("KNN CM\n" ,confusion_matrix(y_test, knn_pred))
    print ("SVM CM\n" ,confusion_matrix(y_test, svm_pred))
    print ("KSVM -rbf CM\n" ,confusion_matrix(y_test, ksvm_pred))
    print ("Navie Bayes CM\n" ,confusion_matrix(y_test, nb_pred))
    print ("DT CM\n" ,confusion_matrix(y_test, dt_pred))
    print ("RF CM\n" ,confusion_matrix(y_test, rf_pred))

def getnullvaluecount(dataset):
    print("--------Start dataset-----------")
    for i in dataset.columns :
        print (i, ":\t\t\t\t", sum(dataset[i].isnull()))
    print("--------End dataset-----------\n")


def printValueCount(dataset, max_cols=5):
    pp = pprint.PrettyPrinter(indent=4)
    map1 = {}
    for i in dataset.columns:
        s = []
        s.append(['nan', sum(dataset[i].isnull())])
        count = 0
        for j, v in dataset[i].value_counts(dropna=False).iteritems():
            count = count + 1
            if count > max_cols:
                s.insert(0, "ManyValues >" + str(max_cols))
                break
            s.append([j, v])
        map1[i] = s
    # for i in map1:
    #    print(i, "\t:\t", map1[i])
    print(tabulate(map1, headers=map1.keys()))

def raise_(ex):
    raise ex

def preprocess1(dataset, mf_imputer, labelencoder, delete_rows=True):
    dataset['Gender'] = dataset['Gender'].map(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else x)
    dataset['Gender'] = mf_imputer.fit_transform(dataset[['Gender']]).ravel()
    dataset['Married'] = dataset['Married'].map(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else x)

    dataset['Dependents'] = dataset['Dependents'].map(lambda x: 4 if x == '3+' else x)
    dataset['Dependents'] = pd.to_numeric(dataset['Dependents'], errors='coerce')

    dataset['Self_Employed'] = dataset['Self_Employed'].map(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else x)
    dataset['Property_Area'] = labelencoder.fit_transform(dataset['Property_Area'])
    dataset['Education'] = labelencoder.fit_transform(dataset['Education'])

    dataset['Gender'] = pd.to_numeric(dataset['Gender'], errors='coerce').astype(np.int8)
    dataset['Dependents'] = pd.to_numeric(dataset['Dependents'], errors='coerce').astype(np.int8)

    printValueCount(dataset)
    cols = dataset.columns
    from fancyimpute import (
        BiScaler,
        KNN,
        NuclearNormMinimization,
        SoftImpute,
        SimpleFill
    )

    X_filled_knn = KNN(k=3).complete(dataset)
    X_filled_mean = SimpleFill("mean").complete(dataset)
    X_filled_softimpute = SoftImpute().complete(dataset)

    simplefill_mse = ((X_filled_mean - dataset) ** 2).mean()
    # print("KNN: %f" % simplefill_mse)

    knn_mse = ((X_filled_knn - dataset) ** 2).mean()
    # print("KNN: %f" % knn_mse)

    softImpute_mse = ((X_filled_softimpute - dataset) ** 2).mean()
    # print("SoftImpute MSE: %f" % softImpute_mse)

    dataset = getDataFrame(X_filled_knn, cols)
   # printValueCount(dataset)

    return dataset
   # df.sex = df.sex.map({'female': 1, 'male': 0})

    # dataset['Dependents'] = mf_imputer.fit_transform(dataset[['Dependents']]).ravel()
    # Impute Dependents using married people
    for row, v in dataset['Dependents'].iteritems():
        if (v is np.nan):
            #print(row, dataset.loc[row, 'Married'])
            if (dataset.loc[row, 'Married'] == 0):
                dataset.loc[row, 'Dependents'] = 0
            else:
                dataset.loc[row, 'Dependents'] = 1

    # Impute married missing values using dependents


    dataset['Dependents'] = dataset['Dependents'].map(
        lambda x: int(x) if str(x).isalnum() and not str(x).isalpha() else raise_('Number format exception'))
    for row, v in dataset['Married'].iteritems():
        if (v != v or v is np.nan or v is None or v is '' or v == ""):
            if (dataset.loc[row, 'Dependents'] > 0):
                dataset.loc[row, 'Married'] = 1
            else:
                dataset.loc[row, 'Married'] = 0

    # dataset['Married'].groupby(dataset['Dependents']).value_counts()

    # Impute missing values of Self_Employed

    #plt.figure(figsize=(16, 6))
    #sns.boxplot(x=dataset['Self_Employed'], y=dataset['ApplicantIncome'])
    # plt.yscale("log")
    #plt.title('Self employed wise boxplot of income')
    #plt.xticks(rotation=90);

    #plt.figure(figsize=(16, 6))
    #sns.boxplot(x=dataset['Self_Employed'], y=dataset['CoapplicantIncome'])
    # plt.yscale("log")
    #plt.title('Self employed wise boxplot of CoapplicantIncome')
    #plt.xticks(rotation=90);

    incomemaan = dataset['ApplicantIncome'].groupby(dataset['Self_Employed']).mean()
    for row, v in dataset['Self_Employed'].iteritems():
        if (v != v or v is np.nan or v is None or v is '' or v == ""):
            print(row, v)
            if (dataset.loc[row, 'ApplicantIncome'] > incomemaan[1]):
                dataset.loc[row, 'Self_Employed'] = 1
            else:
                dataset.loc[row, 'Self_Employed'] = 0

    # Impute missing values of LoanAmount
    #f, ax = plt.subplots(1, 2, figsize=(14, 6))
    #ax1, ax2 = ax.flatten()
    #sns.distplot(dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean()), color='r', ax=ax1)
    #ax1.set_title('Distrbution of LoanAmount')
    #sns.boxplot(x=dataset['LoanAmount'], ax=ax2)
    #ax2.set_ylabel('')
    #ax2.set_title('Boxplot of LoanAmount')

    # Remove ouliers for ApplicantIncome remove >8000
    #sns.distplot(dataset['ApplicantIncome'])
    if (delete_rows == True):
        dataset = dataset[dataset['ApplicantIncome'] < 15000]
    #sns.distplot(dataset['ApplicantIncome'])

    # Remove ouliers for LoanAmount remove >500
    #sns.distplot(dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean()))
    if (delete_rows == True):
        dataset = dataset[dataset['LoanAmount'] < 450]
    else:
        dataset['LoanAmount'] = dataset['LoanAmount'].map(lambda x: 100 if x != x or x is None or x is np.nan else x)
    #sns.distplot(dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean()))

    #sns.jointplot(x=dataset['LoanAmount'], y=dataset['Property_Area'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['LoanAmount'], y=dataset['Loan_Amount_Term'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['LoanAmount'], y=dataset['Credit_History'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['LoanAmount'], y=dataset['ApplicantIncome'], color='g')  # Do not have any relation

    #sns.distplot(dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean()))
    #sns.jointplot(x=dataset['Loan_Amount_Term'], y=dataset['Property_Area'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['Loan_Amount_Term'], y=dataset['LoanAmount'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['Loan_Amount_Term'], y=dataset['Credit_History'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['Loan_Amount_Term'], y=dataset['ApplicantIncome'], color='g')  # Do not have any relation

    # It is not significent with any column so make it mean
    dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean(), inplace=True)

    # Impute Credit_History
    #sns.distplot(dataset['Credit_History'].fillna(dataset['Credit_History'].mean()))
    #sns.jointplot(x=dataset['Credit_History'], y=dataset['Property_Area'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['Credit_History'], y=dataset['LoanAmount'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['Credit_History'], y=dataset['Loan_Amount_Term'], color='g')  # Do not have any relation
    #sns.jointplot(x=dataset['Credit_History'], y=dataset['ApplicantIncome'], color='g')  # Do not have any relation

    # It is not significent with any column so make it most_frequent
    dataset['Credit_History'] = mf_imputer.fit_transform(dataset[['Credit_History']]).ravel()
    printValueCount(dataset, 5)
    return dataset