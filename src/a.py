dataset['Gender'] = dataset['Gender'].map(lambda x: 1 if x =='Male' else 0 if x == 'Female' else None)
dataset['Gender'] = mf_imputer.fit_transform(dataset[['Gender']]).ravel()
dataset['Married'] = dataset['Married'].map(lambda x: 1 if x =='Yes' else 0 if x == 'No' else None)
dataset['Dependents'] = dataset['Dependents'].map(lambda x: 4 if x =='3+' else x)
dataset['Self_Employed'] = dataset['Self_Employed'].map(lambda x: 1 if x =='Yes' else 0 if x == 'No' else None)
dataset['Property_Area'] = labelencoder.fit_transform(dataset['Property_Area'])
dataset['Education'] = labelencoder.fit_transform(dataset['Education'])
dataset['Loan_Status'] = labelencoder.fit_transform(dataset['Loan_Status'])


#dataset['Dependents'] = mf_imputer.fit_transform(dataset[['Dependents']]).ravel()
#Impute Dependents using married people
for row,v in dataset['Dependents'].iteritems():
    if (v is np.nan):
        print (row, dataset.loc[row, 'Married'])
        if(dataset.loc[row, 'Married'] == 0):
            dataset.loc[row, 'Dependents'] = 0
        else:
            dataset.loc[row, 'Dependents'] = 1

#Impute married missing values using dependents
def raise_(ex):
    raise ex

dataset['Dependents'] =  dataset['Dependents'].map(lambda x : int(x) if str(x).isalnum() and not str(x).isalpha() else raise_('Number format exception'))
for row,v in dataset['Married'].iteritems():
    if (v != v or v is np.nan or v is None or v is '' or v == ""):
        if(dataset.loc[row, 'Dependents'] > 0):
            dataset.loc[row, 'Married'] = 1
        else:
            dataset.loc[row, 'Married'] = 0


#dataset['Married'].groupby(dataset['Dependents']).value_counts()

#Impute missing values of Self_Employed
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,6))
sns.boxplot(x = dataset['Self_Employed'], y = dataset['ApplicantIncome'])
#plt.yscale("log")
plt.title('Self employed wise boxplot of income')
plt.xticks(rotation=90);

plt.figure(figsize=(16,6))
sns.boxplot(x = dataset['Self_Employed'], y = dataset['CoapplicantIncome'])
#plt.yscale("log")
plt.title('Self employed wise boxplot of CoapplicantIncome')
plt.xticks(rotation=90);



incomemaan = dataset['ApplicantIncome'].groupby(dataset['Self_Employed']).mean()
for row,v in dataset['Self_Employed'].iteritems():
    if (v != v or v is np.nan or v is None or v is '' or v == ""):
        print (row ,v)
        if(dataset.loc[row, 'ApplicantIncome'] > incomemaan[1]):
            dataset.loc[row, 'Self_Employed'] = 1
        else:
            dataset.loc[row, 'Self_Employed'] = 0


#Impute missing values of LoanAmount
f, ax = plt.subplots(1,2,figsize=(14,6))
ax1,ax2 = ax.flatten()
sns.distplot(dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean()),color='r',ax=ax1)
ax1.set_title('Distrbution of LoanAmount')
sns.boxplot(x = dataset['LoanAmount'], ax=ax2)
ax2.set_ylabel('')
ax2.set_title('Boxplot of LoanAmount')

#Remove ouliers for ApplicantIncome remove >8000
sns.distplot(dataset['ApplicantIncome'])
dataset = dataset[dataset['ApplicantIncome']<15000]
sns.distplot(dataset['ApplicantIncome'])

#Remove ouliers for LoanAmount remove >500
sns.distplot(dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean()))
dataset = dataset[dataset['LoanAmount']<450]
sns.distplot(dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean()))


sns.jointplot( x = dataset['LoanAmount'],y = dataset['Property_Area'],color='g') #Do not have any relation
sns.jointplot( x = dataset['LoanAmount'],y = dataset['Loan_Amount_Term'],color='g') #Do not have any relation
sns.jointplot( x = dataset['LoanAmount'],y = dataset['Credit_History'],color='g') #Do not have any relation
sns.jointplot( x = dataset['LoanAmount'],y = dataset['ApplicantIncome'],color='g') #Do not have any relation

sns.distplot(dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean()))
sns.jointplot( x = dataset['Loan_Amount_Term'],y = dataset['Property_Area'],color='g') #Do not have any relation
sns.jointplot( x = dataset['Loan_Amount_Term'],y = dataset['LoanAmount'],color='g') #Do not have any relation
sns.jointplot( x = dataset['Loan_Amount_Term'],y = dataset['Credit_History'],color='g') #Do not have any relation
sns.jointplot( x = dataset['Loan_Amount_Term'],y = dataset['ApplicantIncome'],color='g') #Do not have any relation

#It is not significent with any column so make it mean
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean(), inplace=True)

#Impute Credit_History 
sns.distplot(dataset['Credit_History'].fillna(dataset['Credit_History'].mean()))
sns.jointplot( x = dataset['Credit_History'],y = dataset['Property_Area'],color='g') #Do not have any relation
sns.jointplot( x = dataset['Credit_History'],y = dataset['LoanAmount'],color='g') #Do not have any relation
sns.jointplot( x = dataset['Credit_History'],y = dataset['Loan_Amount_Term'],color='g') #Do not have any relation
sns.jointplot( x = dataset['Credit_History'],y = dataset['ApplicantIncome'],color='g') #Do not have any relation

#It is not significent with any column so make it most_frequent
dataset['Credit_History'] = mf_imputer.fit_transform(dataset[['Credit_History']]).ravel()
util.printValueCount(dataset, 5)