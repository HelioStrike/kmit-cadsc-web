from __future__ import print_function
import numpy as np
from PIL import Image
from fastai.vision import *
from fastai.metrics import error_rate
import cv2
import os
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import h5py
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageDataBunch
from fastai.vision.image import pil2tensor


MODEL_PATH = './pipelines/lymph_node_backend/data/densenet10epochs.pth'


def preprocess(image_location):
    images=[]
    for location in image_location:
        img=cv2.imread(location)
        print("location ",location)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC) #applying truncation 
        rgb = np.repeat(thresh[..., np.newaxis], 3, -1)
        print("type ",type(rgb))
        images.append((Image(pil2tensor(rgb,dtype=np.float32).div_(255))))
        print("tensor ",type(Image(pil2tensor(rgb,dtype=np.float32).div_(255))))
    return images

def predict(images):
    #loadig model
    train_dir="./pipelines/lymph_node/data/data_bunch"
    base_dir="./pipelines/lymph_node/data" #base directory
    l=os.listdir(train_dir)
    #random.shuffle(l) 
    tfms = get_transforms(do_flip=True) 
    #do_flip: if True, a random flip is applied with probability 0.5 to images
    bs=64 # also the default batch size
    print("loaddd")
#ImageDataBunch splits out the imnages (in the train sub-folder) into a training set and validation set (defaulting to an 80/20 percent split)
    data = ImageDataBunch.from_csv(base_dir, ds_tfms=tfms, size=224, suffix=".tiff",folder="data_bunch",csv_labels="dummy_labels.csv", bs=bs)
    print("valid ",data.valid_ds)
    print("train ",data.train_ds)
    print("test ",data.test_ds)
# transform the image values according to the nueral network we are using
    data.normalize(imagenet_stats)

#cnn_learner loads the model into learn variable`
    learn = cnn_learner(data,models.densenet161, metrics=error_rate, callback_fns=ShowGraph) 


    
    learn=learn.load("./densenet10epochs")
    #predicting labels
    print("prediction ",type(images))
    print(images)
    print("size ",len(images),images[0].shape)
    preds=learn.predict(images[0])
    #preds=learn.pred_batch(np.array(images))  #TO-DO!!!!!!!!
    print(type(preds))
    print("prediction ",preds)
    return preds






#pass in the original , and get the image to be displayed on the website
def get_display_image(orig):
    #preprocessing
    print(type(orig))
    print("def init LOCCCCCC",orig)
    preprocessed=preprocess(orig)
    preds=predict(preprocessed)
    print(preds)
    fh = 30
    fw = 15
    f2 = h5py.File("./pipelines/lymph_node/data/camelyonpatch_level_2_split_test_y.h5", 'r') #contains actuals test labels
    set2 = f2['y']
    print(len(set2))
    imgs = [orig,preds]
    titles = ['Image', 'Normal', 'Metastatic','actual']
    f, ax = plt.subplots(len(orig), 4 , figsize=(fh,fw))
    print(preds[0])
    for i in range(len(orig)):
        print(preds[0])
        r=int(preds[0])
        print("rrrrrrr ",r)
        """fig = plt.figure()
        fig.set_figheight(3)
        fig.set_figwidth(3)"""
        image_name=orig[i].split("/")
        val=image_name[1].split(".")
        print("val ",val)
        print(set2)
        ans=""       

        if(set2[(int)(val[0])][0][0][0]==0):
                actualresult="Normal"
        else:
                actualresult="Metastatic"
        if(r==0):
                result="Normal"
        else:
                result="Metastatic"
        #plt.imshow(cv2.imread(orig[i]))
        #plt.show()
        m1=str(round(float(preds[2][1])*100,2))
        m2=str(round(float(preds[2][0])*100,2))
        ans=actualresult+" "+"Metastatic"+m1+" "+"Normal"+m2
        print(ans)
        """print("Predicted Percentages")
        print("Normal:",float(preds[2][0])*100,"%","Metastatic:",float(preds[2][1])*100,"%")
        print("\n")
        print("Ground Truth:",actualresult)
        print("Predicted:",result)"""
         
    return ans




