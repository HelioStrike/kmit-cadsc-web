
from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np
import random 
import h5py
import cv2 as cv2
import os

train_dir="./pipelines/lymph_node/data/data_bunch"
base_dir="./pipelines/lymph_node/data" #base directory


l=os.listdir(train_dir)
random.shuffle(l) #shuffle training images





tfms = get_transforms(do_flip=True) #do_flip: if True, a random flip is applied with probability 0.5 to images





bs=64 # also the default batch size
data = ImageDataBunch.from_csv(
    base_dir, 
    ds_tfms=tfms, 
    size=224, 
    suffix=".tif",
    folder="data_bunch",
    #csv_labels="train_labels.csv", 
    csv_labels="dummy_labels.csv", 
    bs=bs)
#ImageDataBunch splits out the imnages (in the train sub-folder) into a training set and validation set (defaulting to an 80/20 percent split)





data.normalize(imagenet_stats)
# transform the image values according to the nueral network we are using





learn = cnn_learner(data,models.densenet161, metrics=error_rate, callback_fns=ShowGraph)
#cnn_learner loads the model into learn variable`






f2 = h5py.File("./pipelines/lymph_node_backend/camelyonpatch_level_2_split_test_y.h5", 'r')
set2 = f2['y']





learn=learn.load("./pipelines/lymph_node/model/densenet10epochs.pth")




import matplotlib.pyplot as plt


    image=orig
    actual=image
    preds=learn.predict(actual)
    r=int(preds[0])
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(3)
    if(set2[i][0][0][0]==0):
        actualresult="Normal"
    else:
        actualresult="Metastatic"
    if(r==0):
        result="Normal"
    else:
        result="Metastatic"
    plt.imshow(color)
    plt.show()
    print("Predicted Percentages")
    print("Normal:",float(preds[2][0])*100,"%","Metastatic:",float(preds[2][1])*100,"%")
    print("\n")
    print("Ground Truth:",actualresult)
    print("Predicted:",result)
    
 
    if(result==set2[i][0][0][0]):
        c=c+1


# In[16]:


actual=open_image("/home/abhay/environments/histopathologic-cancer-detection/new_histotest/"+"0.tiff")
predict=learn.predict(actual)


# In[17]:


predict


# In[20]:


float(predict[2][0])*100


# In[ ]:


print("Testing Accuracy")
print((c/32768)*100)


# In[ ]:




