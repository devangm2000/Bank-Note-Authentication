#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("BankNote_Authentication.csv")


# In[4]:


df.head()


# In[22]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[25]:


X.head(10)


# In[28]:


y.head(10)


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)


# In[30]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


# In[31]:


y_pred=classifier.predict(X_test)


# In[38]:


#Accuracy using confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[39]:


accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)


# In[42]:


#Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)


# In[43]:


#Pickle file
import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()


# In[45]:


#prediction
print(classifier.predict([[2,3,4,1]]))


# In[ ]:




