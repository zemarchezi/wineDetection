#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense 
import numpy as np
from scipy.stats import skew
from scipy.stats import boxcox
from collections import OrderedDict
from sklearn import preprocessing
from tukey_outliers_helper import TukeyOutliersHelper
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge


# # Load Data...

# In[2]:


wines = pd.read_csv('winequality.csv', delimiter=';')


# In[3]:


wines.head()


# In[4]:


wines.info()


# ## Correct the missing or problem values

# In[5]:


def correctValues(x):
    if len(x) > 5:
        return (np.nan)
    else:
        return(pd.to_numeric(x))
def coloWine(x):
    if x == 'White':
        return(0)
    else:
        return (1)


# In[6]:


aa = wines['alcohol'].apply(correctValues)
wines['alcohol'] = aa.values
wines.dropna(inplace=True)


# In[7]:


wines.info()


# In[8]:


wines.describe()


# In[ ]:





# ## Set an column representing the type of wine

# In[10]:


wines['color'] = wines['type'].apply(coloWine)


# In[11]:


wines.head()


# # Graph some distributions...

# In[13]:


redWines = wines[wines['type'] == 'Red']
whiteWines = wines[wines['type'] == 'White']
dfPLot = wines.drop(['type', 'color'], axis = 1)
fig = plt.figure(figsize=(16,24))
features = dfPLot.columns

for i in range(len(features)):
    ax1 = fig.add_subplot(4,3,i+1)
    sns.distplot(whiteWines[features[i]], label='White')
    sns.distplot(redWines[features[i]], label='Red')
plt.legend()


# In[14]:


dfPLot = wines.drop(['type', 'color'], axis = 1)
fig = plt.figure(figsize=(16,24))
features = dfPLot.columns

for i in range(len(features)):
    ax1 = fig.add_subplot(4,3,i+1)
    sns.barplot(x='quality', y=features[i],data=wines, hue='type')


# In[15]:


relationship = sns.pairplot(redWines, vars=['fixed acidity','volatile acidity','citric acid'], hue='quality')
plt.show(relationship)


# In[16]:


relationship = sns.pairplot(whiteWines, vars=['fixed acidity','volatile acidity','citric acid'], hue='quality')
plt.show(relationship)


# In[17]:


plt.figure(figsize=(14,14))
sns.heatmap(wines.iloc[:,0:-1].corr(), cbar = True,  square = True, annot=True)


#  - Nos vinhos tintos, a acidez volátil decresce com a qualidade
#  - Ambos tinto e branco, os cloridos, que se resume em a quantidade de sal, decrescem com a qualidade
#  - Há um aumento da quantidade de álcool com o aumento da qualidade em ambos os tipos de vinho
#  - os níveis de acido cítrico também aumentam com a qualidade para os vinhos tintos
#  - Os sulfatos apresentam um valor maior para vinhos tintos com qualidade maior

# In[89]:


XX = redWines.drop(labels=['quality', 'type', 'color'], axis=1)
XX.shape


# In[91]:


yy = redWines['quality']
available_classes = np.unique(yy)


# In[175]:


seed = 160184
random_state = np.random.RandomState(seed=seed)
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2, random_state=32)


# In[176]:


from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import f1_score
models = OrderedDict()

def score_classifier(y_true, y_pred):
    return f1_score(
        y_true, y_pred,
        average='macro', #Calculate metrics for each label, and find their unweighted mean.
        #This does not take label imbalance into account.
    )


# ## Gaussian Naive Bayes

# In[177]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


# In[178]:


preds = model.predict(X_test)


# In[179]:


plot_confusion_matrix(y_true=y_test, y_pred=preds, normalized=True, classes=available_classes)
plt.show()


# In[180]:


print ("score of gaussian naive bayes")
score_classifier(y_test, preds)


# In[181]:


models['naive_bayes'] = model


# ## Quadratic Discriminant Analysis

# In[182]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[183]:


model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)


# In[184]:


preds = model.predict(X_test)


# In[185]:


print ("score for QDA")
score_classifier(y_true=y_test, y_pred=preds)


# In[186]:


models['qda'] = model


# In[187]:


plot_confusion_matrix(y_true=y_test, y_pred=preds, normalized=True, classes=available_classes)
plt.show()


# ## Logistic Regression

# In[188]:


from sklearn.linear_model import LogisticRegression


# In[189]:


model = LogisticRegression(C=0.05,solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)


# In[192]:


print ("score for Logistic Regression")
score_classifier(y_true=y_test, y_pred=preds)


# In[193]:


models['logistic_regression'] = model


# In[194]:


plot_confusion_matrix(y_true=y_test, y_pred=preds, normalized=True, classes=available_classes)
plt.show()


# In[195]:


models.keys()


# In[196]:


get_ipython().run_cell_magic('time', '', "for key, model in models.items():\n    print (key)\n    inputs, targets = (X_train, y_train) if key == 'knn' else \\\n        (X_train, y_train)\n    model.fit(inputs, targets)")


# In[197]:


score_dict = OrderedDict([( key, score_classifier(
                y_true=y_test, y_pred=model.predict(X_test) if key == 'knn' else model.predict(X_test)) )
                          for (key, model) in models.items()])
score_dict


# In[198]:


df = pd.DataFrame(index=score_dict.keys(), data=score_dict.values(), columns=['score'])
df.shape


# In[199]:



plt.figure(figsize=(15,8))
sns.barplot(x=df.index, y=df['score'])
plt.ylabel('unweighted mean of F1 score per class')
plt.xlabel('models')
plt.title('Comparing Classification Models')
plt.show()

