#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle


# In[68]:


import torch
import cv2


# In[69]:


import tensorflow as tf


# ### Yolo model

# In[80]:


# model = tf.keras.models.load_model("model/")
# model = pickle.load(open('linear_model.sav','rb'))
model = pickle.load(open('nnregessor_model.sav','rb'))
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes=0 # pedestrians


# In[81]:


print(model)


# In[82]:


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2


# ## functions

# ![image.png](attachment:image.png)
# https://docs.nvidia.com/tao/archive/tlt-20/tlt-user-guide/text/preparing_data_input.html

# ![image.png](attachment:image.png)
# https://github.com/ultralytics/yolov5/blob/master/models/common.py

# In[83]:


def get_inputs(results):
    try:
        inputs = []
        out = results.xyxy[0].numpy()
#         out = results.xywh[0].numpy()
        for obj in out:
            if obj[-1] == 0:
                inputs.append(obj[0:4])
                
        return inputs
    
    except:
        print('Empty')


# In[84]:


def get_label_info(file):
    labels = []
    with open(file) as f:
        content = f.readlines()
    for obj in content:
        labels.append(obj.split(' '))
    return labels
    


# In[85]:


def match(results, labels):
    try:
        yolo_xyxy = []
        yolo_xywh = get_inputs(results)
        label_xyxy = []
        votes = {}
        matches = {}
        param_label = []
        all_obj_xyxy = results.xyxy[0].numpy()
        for obj in all_obj_xyxy:
            if obj[-1] == 0:
                yolo_xyxy.append(obj[0:4])
                
        for obj in labels:
            if obj[0] == 'Pedestrian':
                label_xyxy.append(obj[4:8])
                
        label_xyxy = np.array(label_xyxy).astype(float)
        majority = len(label_xyxy)/2 if len(label_xyxy) > len(yolo_xyxy)/2 else len(yolo_xyxy)/2
        
        for ind, obj in enumerate (yolo_xyxy):
            vote ={}
            for inner_ind, val in enumerate (obj):
                label_ind = np.abs(label_xyxy[:,inner_ind] - val).argmin()
                vote[label_ind] = vote.get(label_ind,0)+1
            h_vote = max(vote, key=vote.get)
            if vote[h_vote] > majority:
                matches[ind] = h_vote
        for k,v in matches.items():
            temp = np.array(yolo_xyxy[k])
#             temp = np.array(yolo_xywh[k])
            temp= np.append(temp, labels[v][-2])
            if k == 0:
                param_label.append(temp.astype(float))#.astype(float)
            else:
                param_label = np.vstack((param_label,temp.astype(float))) 
        return param_label
    except:
        pass


# In[86]:


def display_pred(df,img_file,label):
    prediction = model.predict(df) ## change to model.predict(df)[0][0] for ssequential
    xmin , ymin , xmax,ymax = df.iloc[0]
    image = cv2.imread(img_file)
    image = cv2.putText( image,"Prediction %.2f Actual  %.2f" %(prediction,label), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# In[87]:


train_img_dir = 'data_object_image_2/training/image_2/'
train_label_dir = 'data_object_label_2/training/label_2/'
columns = ['xmin','ymin','xmax','ymax']

inputs_labels = []
counter = 0
for filename in os.listdir(train_img_dir):
    if counter == 100:
        break
    counter +=1
    ind = filename[0:6]
    img_file = train_img_dir + ind + '.png'
    label_file = train_label_dir + ind + '.txt'
    results = yolo_model(img_file)
    labels = get_label_info(label_file)
    out = np.array(match(results, labels)) 
    try:
        inputs = out[:, 0:4]
        label = out[:,-1]
        df = pd.DataFrame (inputs,columns= columns)
        display_pred(df,img_file,label)
    except:
            pass


# In[78]:


# np.save('yolo_inputs',inputs_labels[:, 0:4])
# np.save('yolo_labels',inputs_labels[:,-1])


# In[79]:


# np.save('yolo_inputs_labels',inputs_labels)


# In[ ]:




