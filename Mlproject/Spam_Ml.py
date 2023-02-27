#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[7]:


df=pd.read_csv("D:/Mlproject/spam.csv",encoding="latin-1")


# In[6]:


#visualizing dataset
df.head(n=10)


# In[9]:


df.shape


# In[11]:


#to check whether target attribute is binary or not
np.unique(df['class'])


# In[13]:


np.unique(df['message'])


# In[14]:


#createing sparse matrix
x=df["message"].values #datafreme targeting message column
y=df["class"].values
#create count vectorizer object
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()

print(v)


# In[15]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[16]:


#spliting train + test 3:1
train_x=x[:4180]
train_y=y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[18]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)

y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[20]:


#training score
print(bnb.score(train_x,train_y)*100)

#print score
print(bnb.score(test_x,test_y)*100)


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))

