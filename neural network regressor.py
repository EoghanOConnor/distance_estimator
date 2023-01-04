#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
import os
import torch
import cv2
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import pandas as pd
import time
import pickle


# In[19]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


# In[20]:


inputs_labels = pd.read_csv('pedestrians.csv')


# In[21]:


inputs = inputs_labels[['xmin', 'ymin', 'xmax', 'ymax']]


# In[22]:


labels = inputs_labels['zloc']


# In[23]:


inputs


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.10, random_state=42)


# In[25]:


regr =  MLPRegressor(random_state=1, max_iter=500)
  
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


# In[26]:


print(X_test)


# In[27]:


prediction = regr.predict(X_test)


# In[28]:


for pred, true in zip (prediction, y_test):
    print(f"Prediction ; {pred} true value: {true}")


# In[29]:


mean_absolute_error(y_test,prediction)


# In[30]:


r2_score(y_test,prediction)


# In[31]:


filename = 'nnregessor_model.sav'
pickle.dump(regr,open(filename,'wb'))


# make a Neural Network regressor next.
# test using the quick model test.
