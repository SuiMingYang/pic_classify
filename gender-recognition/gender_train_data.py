#coding=utf-8
 
import os
import numpy as np
from cv2 import cv2


def get_img_list(dirname,flag=0):
    rootdir= os.path.abspath(url+dirname+'/')
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    files=[]
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            files.append(path)
    return files


images=[]
labels=[]

def read_img(list,flag=0):
    for i in range(len(list)-1):
         if os.path.isfile(list[i]):
             images.append(cv2.imread(list[i]).flatten())
             labels.append(flag)

labels_text=[]
url = './gender-recognition/data/train/female'
dr = os.listdir(url)
#dr.remove('.DS_Store')
for i,d in enumerate(dr):
    if os.path.isdir(url+d):
        #files.append(d)
        flag=[0]*len(dr)
        flag[i]=1
        read_img(get_img_list(d),flag)
        labels_text.append(d)
        #read_img(get_img_list(''),[1,0])



images = np.array(images)
labels = np.array(labels)

#重新打乱
permutation = np.random.permutation(labels.shape[0])
all_images = images[permutation,:]
all_labels = labels[permutation,:]

#训练集与测试集比例 8：2
train_total = all_images.shape[0]
train_nums= int(all_images.shape[0]*0.8)
test_nums = all_images.shape[0]-train_nums

images = all_images[0:train_nums,:]
labels = all_labels[0:train_nums,:]


test_images = all_images[train_nums:train_total,:]
test_labels = all_labels[train_nums:train_total,:]


