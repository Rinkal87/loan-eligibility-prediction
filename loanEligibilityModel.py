#Importing neccessary lib...

import numpy as np
import pandas as pd

# Loading the dataset

loan_train = pd.read_csv('loan-train.csv')
loan_test = pd.read_csv('loan-test.csv')
conert_dict={
    'ApplicantIncome': np.float16,
    'CoapplicantIncome': np.float16 ,
    'LoanAmount' : np.float16 ,
    'Loan_Amount_Term' : np.float16 ,
    'Credit_History'  : np.float16
}
loan_train=loan_train.astype(conert_dict)

#  Explore Feature having Object Data Type

def explore_object(data,feature):

    if(data[feature].dtype == 'object'):

        print(data[feature].value_counts())
        print()

col=[col for col in loan_train.columns]
for col in col:
    explore_object(loan_train,col)

# Now Removing Outliers
q1,q3=loan_train['LoanAmount'].quantile(0.25),loan_train['LoanAmount'].quantile(0.75)
iqr=q3-q1
lower_limit,higher_limit=q1-(1.5*iqr),q3+(1.5*iqr)
lower_limit,higher_limit
loan_train=loan_train[loan_train['LoanAmount']<higher_limit]
mode_credit_history = loan_train.loc[:,'Credit_History'].mode()
loan_train.loc[:,'Credit_History'] = loan_train.loc[:,'Credit_History'].fillna(mode_credit_history.values[0]) #Mode

mode_credit_history = loan_test.loc[:,'Credit_History'].mode()
loan_test.loc[:,'Credit_History'] = loan_test.loc[:,'Credit_History'].fillna(mode_credit_history.values[0])   #Mode


mode_loan_term = loan_train.loc[:,'Loan_Amount_Term'].mode()
loan_train.loc[:,'Loan_Amount_Term'] = loan_train.loc[:,'Loan_Amount_Term'].fillna(mode_loan_term.values[0])   #Mode

mode_loan_term = loan_test.loc[:,'Loan_Amount_Term'].mode()
loan_test.loc[:,'Loan_Amount_Term'] = loan_test.loc[:,'Loan_Amount_Term'].fillna(mode_loan_term.values[0])   #Mode


mean_loan_amount = loan_train.loc[:,'LoanAmount'].mean()
loan_train.loc[:,'LoanAmount'] = loan_train.loc[:,'LoanAmount'].fillna(mean_loan_amount)   #Mean

mean_loan_amount = loan_test.loc[:,'LoanAmount'].mean()
loan_test.loc[:,'LoanAmount'] = loan_test.loc[:,'LoanAmount'].fillna(mean_loan_amount)   #Mean
# We need to fill null values of Catagorical Data with mode

mode_gender = loan_train.loc[:,'Gender'].mode()
loan_train.loc[:,'Gender'] = loan_train.loc[:,'Gender'].fillna(mode_gender.values[0])   #Mode

mode_gender = loan_test.loc[:,'Gender'].mode()
loan_test.loc[:,'Gender'] = loan_test.loc[:,'Gender'].fillna(mode_gender.values[0])   #Mode



mode_married = loan_train.loc[:,'Married'].mode()
loan_train.loc[:,'Married'] = loan_train.loc[:,'Married'].fillna(mode_married.values[0])   #Mode

mode_married = loan_test.loc[:,'Married'].mode()
loan_test.loc[:,'Married'] = loan_test.loc[:,'Married'].fillna(mode_married.values[0])   #Mode

mode_dependents = loan_train.loc[:,'Dependents'].mode()
loan_train.loc[:,'Dependents'] = loan_train.loc[:,'Dependents'].fillna(mode_dependents.values[0])   #Mode

mode_dependents = loan_test.loc[:,'Dependents'].mode()
loan_test.loc[:,'Dependents'] = loan_test.loc[:,'Dependents'].fillna(mode_dependents.values[0])   #Mode



mode_self_employed = loan_train.loc[:,'Self_Employed'].mode()
loan_train.loc[:,'Self_Employed'] = loan_train.loc[:,'Self_Employed'].fillna(mode_self_employed.values[0])   #Mode

mode_self_employed = loan_test.loc[:,'Self_Employed'].mode()
loan_test.loc[:,'Self_Employed'] = loan_test.loc[:,'Self_Employed'].fillna(mode_self_employed.values[0])   #Mode

# Removing Unnecessary Features

loan_train = loan_train.drop('Loan_ID',axis=1)
X = loan_train.iloc[:,:-1]
y = loan_train['Loan_Status']
X = pd.get_dummies(X)
loan_train = pd.get_dummies(loan_train)
loan_test = pd.get_dummies(loan_test)
# Splitting the data set into train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn import svm

model_4=svm.SVC(kernel='linear')
model_4.fit(X_train,y_train)
model_4_prediction=model_4.predict(X_test)


import joblib

# Save the trained model
joblib.dump(model_4, 'loan_eligibility_model.pkl')
