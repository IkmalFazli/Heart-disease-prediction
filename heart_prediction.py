# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:07:54 2022

@author: Si Kemal
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Colab Notebooks

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.stats as ss
import pandas as pd
import numpy as np
import pickle
import os

def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


"""# EDA
Static
"""

DATA_PATH = os.path.join(os.getcwd(),'heart.csv')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

"""# STEP 1) DATA LOADING"""

df = pd.read_csv(DATA_PATH)

"""# STEP 2) DATA INSPECTION"""

print(df.head(10))
print(df.tail(10))
df.info()
df.describe().T

plt.figure(figsize=(20,20))
df.boxplot()
plt.show()

Categorical = ['sex', 'cp','fbs','restecg','exng','slp','caa','thall','output']
Continuous = ['age','trtbps','chol','thalachh','oldpeak']

import matplotlib.pyplot as plt
import seaborn as sns


for con in Continuous:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

for cat in Categorical:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()

df.duplicated().sum()

"""# STEP 3) DATA CLEANING
- no null
- remove duplicated in the data."""

df=df.drop_duplicates()
df.duplicated().sum()

df.info()

"""# STEP 4) FEATURE SELECTION
"""

for i in Continuous:
    LR = LogisticRegression(solver='liblinear')
    LR.fit(np.expand_dims(df[i],axis=-1),df['output'])
    print(i+' '+str(LR.score(np.expand_dims(df[i],axis=-1),df['output'])))

cate =  ['sex', 'cp','fbs','restecg','exng','slp','caa','thall']
for j in cate:
    confussion_mat = pd.crosstab(df[j],df['output']).to_numpy()
    print(j + ' ' + str(cramers_corrected_stat(confussion_mat)))
    

"""# STEP 5) PREPROCESSING
     based on the result from both test, I choose 'age','trtbps','chol',
    'thalachh','oldpeak' for continuous features since the score for 
    all of them more than 50% and for categorical variable only the highest 
    were chosen which is 'cp' and 'thall'"""

from sklearn.model_selection import train_test_split

x = df[['age','trtbps','chol','thalachh','oldpeak','cp','thall']]
        
y = df['output']

X_train,X_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.3,
                                                 random_state=238)

#Steps for Standard scaler
step_LR_ss = Pipeline([('Standard Scaler',StandardScaler()),
           ('Logistic Regression Classifier', LogisticRegression(solver='liblinear'))]) #Standard Scaler

# Steps for Min-Max Scaler
step_LR_mms = Pipeline([('Min Max Scaler',MinMaxScaler()),
             ('Logistic Regression Classifier', LogisticRegression(solver='liblinear'))])

#Steps for Standard scaler
step_RF_ss = Pipeline([('Standard Scaler',StandardScaler()),
           ('Random Forest Classifier',RandomForestClassifier())]) #Standard Scaler

# Steps for Min-Max Scaler
step_RF_mms = Pipeline([('Min Max Scaler',MinMaxScaler()),
             ('Random Forest Classifier',RandomForestClassifier())])

#Steps for Standard scaler
step_DT_ss = Pipeline([('Standard Scaler',StandardScaler()),
           ('Decision Tree Classifier',DecisionTreeClassifier())]) #Standard Scaler

# Steps for Min-Max Scaler
step_DT_mms = Pipeline([('Min Max Scaler',MinMaxScaler()),
             ('Decision Tree Classifier',DecisionTreeClassifier())])

#Steps for Standard scaler
step_KNN_ss = Pipeline([('Standard Scaler',StandardScaler()),
           ('KNeighbors Classifier',KNeighborsClassifier())]) #Standard Scaler

# Steps for Min-Max Scaler
step_KNN_mms = Pipeline([('Min Max Scaler',MinMaxScaler()),
             ('KNeighbors Classifier',KNeighborsClassifier())])

#Steps for Standard scaler
step_SVC_ss = Pipeline([('Standard Scaler',StandardScaler()),
           ('SVC Classifier',SVC())]) #Standard Scaler

# Steps for Min-Max Scaler
step_SVC_mms = Pipeline([('Min Max Scaler',MinMaxScaler()),
             ('SVC Classifier',SVC())])

# Create a list for the pipeline so that you can iterate them
pipelines = [step_LR_ss,step_LR_mms,step_RF_ss,step_RF_mms,
             step_DT_ss,step_DT_mms,step_KNN_ss,step_KNN_mms,
             step_SVC_ss,step_SVC_mms]

#Fitting of data
for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_dict = {0:'Logistic Regression with Standard Scaler Approach', 
             1:'Logistic Regression with Min-max Scaler Approach',
             2:'Random Forest with Standard Scaler Approach', 
             3:'Random Forest with Min-max Scaler Approach',
             4:'Decision Tree with Standard Scaler Approach', 
             5:'Decision Tree with Min-max Scaler Approach',
             6:'KNeighbors with Standard Scaler Approach',
             7:'KNeighbors with Min-max Scaler Approach',
             8:'SVC with Standard Scaler Approach',
             9:'SVC with Min-max Scaler Approach'}

best_accuracy = 0

# model evaluation
for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]
        
print('The best modeling and scaling approach for Heart.csv train Dataset will be {} with accuracy of {}'.format(best_scaler,best_accuracy))

""" Referring to the result it is clear that Logistic Regression with MinMax 
    scaler has the highest score which is 0.835. Hence we pick model to best 
    tested further to find the best parameter"""

step_LR_mms = Pipeline([('Min-Max Scaler',MinMaxScaler()),
                       ('classifier',LogisticRegression())])

grid_params =[{'classifier' : [LogisticRegression()],
               'classifier__C' :[100,1,0.1,0.01],
               'classifier__solver' : ['liblinear']}]

gridsearch = GridSearchCV(step_LR_mms, grid_params,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

""" As the result for gridsearch parameter, clearly the default parameter is 
    the best parameter for this case. Hence we save the model for deployment"""

best_model = Pipeline([('Min-Max Scaler',MinMaxScaler()),
                       ('classifier',LogisticRegression(solver='liblinear'))])


best_model.fit(X_train,y_train)

with open(BEST_MODEL_PATH,'wb') as file:
  pickle.dump(best_model,file)

y_true = y_test
y_predict = best_model.predict(X_test)

class_report = classification_report(y_true, y_predict)
confusion_mat = confusion_matrix(y_true,y_predict)
acc_score = accuracy_score(y_true, y_predict)

print(class_report)
print(confusion_mat)
print(acc_score)

disp = ConfusionMatrixDisplay(confusion_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
disp.plot(cmap=plt.cm.Blues)
plt.show()


