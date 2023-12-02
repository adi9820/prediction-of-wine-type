#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')
white


# In[3]:


# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')
red


# In[4]:


red.describe()


# In[5]:


white.describe()


# In[6]:


red.isnull().sum()


# In[7]:


white.isnull().sum()


# In[8]:


fig, ax = plt.subplots(1, 2)
 
ax[0].hist(red["alcohol"], 10, facecolor ='red', alpha = 0.5, label ="Red wine")
 
ax[1].hist(white["alcohol"], 10, facecolor ='white', ec ="black", lw = 0.5, alpha = 0.5, label ="White wine")
 
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
 
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()


# In[9]:


red["type"]=1
white["type"]=0
wines=pd.concat([red, white], ignore_index=True)


# In[10]:


wines


# In[11]:


plt.figure(figsize = (12,10))
sns.heatmap(wines.corr(), annot = True, fmt = ".2f", cmap = "YlGnBu")
plt.title("correleation heatmap")
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split
x=wines.drop("type", axis=1)
y=np.array(wines["type"])


# In[13]:


x


# In[14]:


y


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.76, random_state=45)


# In[16]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()

model.add(Dense(12, activation ='relu', input_shape =(12, )))
model.add(Dense(9, activation ='relu'))
model.add(Dense(1, activation ='sigmoid'))

model.output_shape
model.summary()
model.get_config()

model.get_weights()

model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics =['accuracy'])


# In[17]:


model.fit(x_train, y_train, epochs = 4, batch_size = 1, verbose = 1)

y_pred = model.predict(x_test)
print(y_pred)


# In[18]:


classified_ypred = [1 if pred >= 0.5 else 0 for pred in y_pred]


# In[19]:


classified_ypred


# In[22]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, classified_ypred)
print("F1 Score:", f1)


# In[ ]:




