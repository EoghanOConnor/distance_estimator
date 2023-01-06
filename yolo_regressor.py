#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import pandas as pd 


# In[2]:


import torch
import cv2
import tensorflow as tf


# In[3]:


yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes=0 # pedestrians


# In[4]:


def get_inputs(results):
    try:
        inputs = []
        out = results.xyxy[0].numpy()
        for obj in out:
            if obj[-1] == 0:
                inputs.append(obj[0:4])
                
        return inputs
    
    except:
        print('Empty')


# In[5]:


train_img_dir = 'data_object_image_2/training/image_2/'
train_label_dir = 'data_object_label_2/training/label_2/'


# In[6]:


img_file = train_img_dir + '000000' + '.png'
results = yolo_model(img_file)


# In[7]:


inputs = get_inputs(results)


# In[8]:


columns = ['xmin','ymin','xmax','ymax']
df = pd.DataFrame (inputs,columns= columns)


# In[9]:


print(inputs[0])


# In[10]:


print(df)


# In[11]:


model = tf.keras.models.load_model("model/")


# In[12]:


out = model.predict(df)[0][0]


# In[13]:


xmin , ymin , xmax,ymax = df.iloc[0]


# In[14]:


print(out)


# In[15]:


print(df.iloc[0])


# In[16]:


label_file = train_label_dir + '000000' + '.txt'


# In[17]:


print(label_file)


# In[18]:


def get_label_info(file):
    labels = []
    with open(file) as f:
        content = f.readlines()
    for obj in content:
        labels.append(obj.split(' '))
    return labels


# In[19]:


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2


# In[20]:


# image = cv2.imread(img_file)


# In[21]:


# image = cv2.putText( image,"Prediction %.2f Actual  %.2f" %(out,label), org, font, 
#                    fontScale, color, thickness, cv2.LINE_AA)


# In[22]:


t = 2.12345
x = 1.12345
print("%.2f sfs %.2f" %(t,x))


# In[23]:


int(df['ymax'][0])


# In[24]:


# cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),2)
# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[25]:


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
            temp = np.array(yolo_xywh[k])
            temp= np.append(temp, labels[v][-2])
            if k == 0:
                param_label.append(temp.astype(float))#.astype(float)
            else:
                param_label = np.vstack((param_label,temp.astype(float))) 
        return param_label
    except:
        pass


# In[26]:


train_img_dir = 'data_object_image_2/training/image_2/'
train_label_dir = 'data_object_label_2/training/label_2/'
inputs_labels = []
counter = 0
for filename in os.listdir(train_img_dir):
    if counter == 29:
        break
    counter +=1
    
    ind = filename[0:6]
    img_file = train_img_dir + ind + '.png'
    label_file = train_label_dir + ind + '.txt'
    results = yolo_model(img_file)
    labels = get_label_info(label_file)
    out = np.array(match(results, labels))
#     try:
#         if not len(inputs_labels):
#             inputs_labels = out
#         else:
#             inputs_labels = np.vstack((inputs_labels,out))
#     except:
#         pass


# In[27]:


counter = 0
columns = ['xmin','ymin','xmax','ymax']

for filename in os.listdir(train_img_dir):
    if counter == 5:
        break
    counter +=1
    ind = filename[0:6]
    img_file = train_img_dir + ind + '.png'
    label_file = train_label_dir + ind + '.txt'
    
    #Yolo Model
    results = yolo_model(img_file)
    inputs = get_inputs(results)
    df = pd.DataFrame (inputs,columns= columns)
    
    #Label
    label = get_label_info(label_file)
    
    image = cv2.imread(img_file)
    image = cv2.putText( image,"Prediction %.2f Actual  %.2f" %(out,label), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     try:
#         if not len(inputs_labels):
#             inputs_labels = out
#         else:
#             inputs_labels = np.vstack((inputs_labels,out))
#     except:
#         pass
    


# In[ ]:




