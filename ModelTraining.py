#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("BankNote_Authentication.csv")


# In[3]:


df.head()


# In[4]:


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[5]:


X


# In[6]:


y


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


# In[9]:


y_pred=classifier.predict(X_test)


# In[10]:


#Accuracy using confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[11]:


#Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc


# In[12]:


tp=cm[0][0]
fp=cm[0][1]
tn=cm[1][1]
fn=cm[1][0]
precision=tp/(tp+fp)
precision


# In[13]:


recall=tp/(tp+fn)
recall


# In[14]:


#F1 = 2 * (precision * recall) / (precision + recall)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, zero_division=1)


# In[15]:


#Pickle file
import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()


# In[16]:


#prediction
classifier.predict([[2,3,4,1]])


# In[ ]:




