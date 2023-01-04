#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import torch
import cv2
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import pandas as pd
import time
import pickle


# In[27]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


# In[28]:


inputs_labels = pd.read_csv('pedestrians.csv')


# In[29]:


inputs = inputs_labels[['xmin', 'ymin', 'xmax', 'ymax']]


# In[30]:


labels = inputs_labels['zloc']


# In[31]:


inputs


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.10, random_state=42)


# In[33]:


regr = LinearRegression()
  
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


# In[39]:


print(X_test)


# In[34]:


prediction = regr.predict(X_test)


# In[35]:


for pred, true in zip (prediction, y_test):
    print(f"Prediction ; {pred} true value: {true}")


# In[36]:


mean_absolute_error(y_test,prediction)


# In[37]:


r2_score(y_test,prediction)


# In[38]:


filename = 'linear_model.sav'
pickle.dump(regr,open(filename,'wb'))


# make a Neural Network regressor next.
# test using the quick model test.
