
# coding: utf-8

# In[55]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns
#import the data, specify data types
import pandas as pd


# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import re

import pickle 
#import mglearn
import time


from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


# In[58]:


df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df.info() 


# In[60]:


df.head()


# In[59]:


df = df.drop('Description', axis = 1)


# In[61]:


df.dtypes


# In[62]:


non_cat = [f for f in df.columns if df.dtypes[f] != 'object']
cat = [f for f in df.columns if df.dtypes[f] == 'object']


# In[ ]:


cat


# In[63]:


non_cat.remove('InvoiceDate')
cat.append('InvoiceDate')


# In[64]:


def treat_missing_numeric(df,columns,how = 'mean'):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mean', 'mode', 'median','ffill', numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'mean':
        for i in columns:
            print("Filling missing values with mean for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mean())
            
    elif how == 'mode':
        for i in columns:
            print("Filling missing values with mode for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mode())
    
    elif how == 'median':
        for i in columns:
            print("Filling missing values with median for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].median())
    
    elif how == 'ffill':
        for i in columns:
            print("Filling missing values with forward fill for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(method ='ffill')
    
    elif type(how) == int or type(how) == float:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(how)
    else:
        print("Missing value fill cannot be completed")
    return df


# In[65]:


def treat_missing_categorical(df,columns,how = 'mode'):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mode', any string or numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'mode':
        for i in columns:
            print("Filling missing values with mode for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mode()[0])
    elif type(how) == str:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(how)
    elif type(how) == int or type(how) == float:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(str(how))
    else:
        print("Missing value fill cannot be completed")
    return df


# In[66]:


treat_missing_numeric(df,non_cat,how = 'mean')


# In[67]:


treat_missing_categorical(df,cat,how = 'mode')


# In[68]:


from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(df[non_cat])
df[non_cat] = std_scale.transform(df[non_cat])


# In[69]:


countries = ['Germany','France']
df_l = df[df.Country.isin(countries)]
df_l.shape 


# In[73]:


df_l.head()


# In[72]:


df_l.loc[ df_l.Country == 'Germany', 'Country'] = 0
df_l.loc[ df_l.Country == 'France', 'Country'] = 1


# In[75]:


cat.remove('Country')


# In[77]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

for col in cat:
    df_l[col]=labelEnc.fit_transform(df_l[col])
    

df_l.head()


# In[78]:


X = df_l.copy()
X = X.drop('Country', axis = 1)
y = df_l['Country']


# In[79]:


def holdout_cv(X,y,size = 0.3, seed = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = seed)
    return X_train, X_test, y_train, y_test


# In[80]:


X_train, X_test, y_train, y_test = holdout_cv(X, y, size = 0.3, seed = 1)


# In[81]:


'''See the dimensions of input and output data set.'''
print('Input Matrix Dimension:  ', X_train.shape)
print('Output Vector Dimension: ', y_train.shape)
print('Test Data Dimension:     ', X_test.shape)


# In[84]:


from sklearn.svm import SVC


# In[85]:


### Running SVM
def runSVC(train_X, train_y, test_X, test_y, C=1.0, kernel_choice = 'rbf'):
    model = SVC(C=C, kernel=kernel_choice, probability=True)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, model


# In[86]:


pred_svm , loss_svm, model_svm = runSVC(X_train, y_train, X_test, y_test)


# In[89]:


pred_svm


# In[88]:


for i in range(len(pred_svm)):
    if pred_svm[i] >= 0.5:
       pred_svm[i] = 1
    else:
        pred_svm[i] = 0


# In[91]:


from sklearn.metrics import accuracy_score


# In[92]:


#pred_test_y_xg
accuracy_score(y_test, pred_svm)


# In[95]:


from sklearn.tree import DecisionTreeClassifier


# In[93]:


### Running Decision Tree
def runDT(train_X, train_y, test_X, test_y, criterion='gini', depth=None, min_split=2, min_leaf=1):
    model = DecisionTreeClassifier(criterion = criterion, max_depth = depth, 
                                   min_samples_split = min_split, min_samples_leaf=min_leaf)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, model


# In[96]:


test_pred_dt, loss_dt, model_dt = runDT( X_train, y_train, X_test, y_test)


# In[97]:


test_pred_dt


# In[98]:


#pred_test_y_xg
accuracy_score(y_test, test_pred_dt)


# In[99]:


from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier


# In[100]:


### Running Random Forest
def runRF(train_X, train_y, test_X, test_y, depth=20, leaf=10, feat=0.2):
    model = RandomForestClassifier(
            n_estimators = 1000,
                    max_depth = depth,
                    min_samples_split = 2,
                    min_samples_leaf = leaf,
                    max_features =  feat,
                    n_jobs = 4,
                    random_state = 0)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, model


# In[101]:


test_pred_rf, loss_rf, model_rf = runRF( X_train, y_train, X_test, y_test)


# In[104]:


for i in range(len(test_pred_rf)):
    if test_pred_rf[i] >= 0.5:
       test_pred_rf[i] = 1
    else:
       test_pred_rf[i] = 0


# In[105]:


#pred_test_y_xg
accuracy_score(y_test, test_pred_rf)


# In[106]:


clas = np.array(['Germany','France'])
clas


# In[109]:


import itertools


# In[107]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("dt Normalized confusion matrix")
    else:
        print('dt Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    


# In[111]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, test_pred_dt)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')


plt.show()


# In[112]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, test_pred_rf)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')


plt.show()


# In[113]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred_svm)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')


plt.show()

