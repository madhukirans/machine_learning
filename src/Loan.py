import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import statsmodels.formula.api as sm
from tabulate import tabulate

from src.mylib import mymllib1 as util
#from src.mylib.mymllib import MyMlUtil


#from importlib import reload
#util = MyMlUtil()

dataset_train = pd.read_csv('/BigData/dat/train.csv')
dataset_test = pd.read_csv('/BigData/dat/test.csv')

dataset_test_id = dataset_test['Loan_ID']
dataset_train.drop('Loan_ID', axis =1, inplace=True)
dataset_test.drop('Loan_ID', axis =1, inplace=True)

mf_imputer = Imputer(missing_values = np.nan, strategy = 'most_frequent', axis = 0)
labelencoder = LabelEncoder()
dataset_train['Loan_Status'] = labelencoder.fit_transform(dataset_train['Loan_Status'])

#util.printValueCount(dataset_train)
dataset_train = util.preprocess1(dataset_train, mf_imputer, labelencoder)


util.printValueCount(dataset_train)
#import seaborn as sns
#import matplotlib.pyplot as plt
#plt.figure(figsize=(16, 6))
#sns.boxplot(x=dataset_train['Self_Employed'], y=dataset_train['ApplicantIncome'])
#sns.distplot(dataset_train['LoanAmount'])
#sns.jointplot(x=dataset_train['LoanAmount'], y=dataset_train['Property_Area'], color='g')  # Do not have any relation

# plt.yscale("log")
#plt.title('Self employed wise boxplot of income')
#plt.xticks(rotation=90);

#plt.figure(figsize=(16, 6))
#sns.boxplot(x=dataset['Self_Employed'], y=dataset['CoapplicantIncome'])
# plt.yscale("log")
#plt.title('Self employed wise boxplot of CoapplicantIncome')
#plt.xticks(rotation=90);

util.printValueCount(dataset_test)
dataset_test =  util.preprocess1(dataset_test, mf_imputer, labelencoder, delete_rows=False)

xx = dataset_train.iloc[:, :-1]
yy = dataset_train.iloc[:, len(dataset_train.columns)-1]



#util.printValueCount(xx)
#xx.mean()

#cols  = util.backwordElimination(xx,yy)
#x_after_bwe = xx.loc[:,cols.index]
#x_test_after_bwe = dataset_test.loc[:,cols.index]

#util.printValueCount(x_after_bwe)
#util.printValueCount(dataset_test)

#x_after_OHE, x_test_after_OHE = util.onehotEncoder(x_after_bwe, x_test_after_bwe, [0, 3, 4])


#Slipt test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.1)

x_train, x_test = util.getPCA(x_train, x_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

x_train = util.featureScaling(x_train)
x_test = util.featureScaling(x_test)

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    #x_set, y_set, classifer = x_train, y_train, model
    x1, x2 = np.meshgrid(np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
                         np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01))
    temp = np.array([x1.ravel(), x2.ravel()]).T
    model.fit(X, y)
    y_pred = model.predict(temp).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, alpha=0.2, cmap=ListedColormap(('red', 'green')))
    plt.xlim(x1.min(), x1.max())
    plt.xlim(x2.min(), x2.max())

    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, i],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.show()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=0)
visualize_classifier(model, x_train, y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
visualize_classifier(model, x_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
visualize_classifier(model, x_train, y_train)

from sklearn.svm import SVC
model = SVC(kernel="poly")
visualize_classifier(model, x_train, y_train)

dataset_test.shape
xx.shape
logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred = util.runAllCategoricalModels(xx, yy, dataset_test)
logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred = util.runAllCategoricalModels(x_train, y_train, x_test)

util.writeCsvFiles(rf_pred, "rf", "Loan_Status" , dataset_test_id)

rf_pred = util.RFClassifer(xx, yy, dataset_test)
#rf_pred = util.logisticClassifer (x3, yy, x_test1)

util.printAccuracy_score(y_test, logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred)
util.printConfusion_matrix(y_test, logis_pred, knn_pred, svm_pred, ksvm_pred, nb_pred, dt_pred, rf_pred)

print( "logistic accuracy_score" , accuracy_score(y_test, logis_pred))
print ("logistic CM\n" ,confusion_matrix(y_test, logis_pred))

print( "KNN accuracy_score" , accuracy_score(y_test, knn_pred))
print ("KNN CM\n" ,confusion_matrix(y_test, knn_pred))

print( "SVM accuracy_score" , accuracy_score(y_test, svm_pred))
print ("SVM CM\n" ,confusion_matrix(y_test, svm_pred))

print( "KSVM -rbf accuracy_score" , accuracy_score(y_test, ksvm_pred))
print ("KSVM -rbf CM\n" ,confusion_matrix(y_test, ksvm_pred))

print( "Navie Bayes accuracy_score" , accuracy_score(y_test, nb_pred))
print ("Navie Bayes CM\n" ,confusion_matrix(y_test, nb_pred))

print( "DT accuracy_score" , accuracy_score(y_test, dt_pred))
print ("DT CM\n" ,confusion_matrix(y_test, dt_pred))

print( "RF accuracy_score" , accuracy_score(y_test, rf_pred))
print ("RF CM\n" ,confusion_matrix(y_test, rf_pred))




x_train = pd.DataFrame(x_train)
x_train.values.ravel
yy.shape