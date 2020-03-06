#-*-coding:utf8-*-#
 
import os
from cv2 import cv2
from PIL import Image,ImageDraw
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import gender_train_data as train_data
from gender_train_data import labels_text
import matplotlib.pyplot as plt

# 人脸检测
class DetectFaces():
    def __init__(self):
        pass
    #detectFaces()返回图像中所有人脸的矩形坐标（矩形左上、右下顶点）
    #使用haar特征的级联分类器haarcascade_frontalface_default.xml，在haarcascades目录下还有其他的训练好的xml文件可供选择。
    #注：haarcascades目录下训练好的分类器必须以灰度图作为输入。
    def detectFaces(self,image_name):
        img = cv2.imread(image_name)
        face_cascade = cv2.CascadeClassifier("./haar/haarcascades/haarcascade_frontalface_default.xml")
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
        result = []
        for (x,y,width,height) in faces:
            result.append((x,y,x+width,y+height))
        return result
    
    
    #保存人脸图
    def saveFaces(self,image_name):
        faces = self.detectFaces(image_name)
        if faces:
            #将人脸保存在save_dir目录下。
            #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
            save_dir = image_name.split('.')[0]+"_faces"
            os.mkdir(save_dir)
            count = 0
            for (x1,y1,x2,y2) in faces:
                file_name = os.path.join(save_dir,str(count)+".jpg")
                Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name)
                count+=1
    
    
    #在原图像上画矩形，框出所有人脸。
    #调用Image模块的draw方法，Image.open获取图像句柄，ImageDraw.Draw获取该图像的draw实例，然后调用该draw实例的rectangle方法画矩形(矩形的坐标即
    #detectFaces返回的坐标)，outline是矩形线条颜色(B,G,R)。
    #注：原始图像如果是灰度图，则去掉outline，因为灰度图没有RGB可言。drawEyes、detectSmiles也一样。
    def drawFaces(self,image_name):
        faces = self.detectFaces(image_name)
        if faces:
            img = Image.open(image_name)
            draw_instance = ImageDraw.Draw(img)
            for (x1,y1,x2,y2) in faces:
                draw_instance.rectangle((x1,y1,x2,y2), outline=(255, 0,0))
            img.save(image_name)
 
 
 

    #检测眼睛，返回坐标
    #由于眼睛在人脸上，我们往往是先检测出人脸，再细入地检测眼睛。故detectEyes可在detectFaces基础上来进行，代码中需要注意“相对坐标”。
    #当然也可以在整张图片上直接使用分类器,这种方法代码跟detectFaces一样，这里不多说。
    def detectEyes(self,image_name):
        eye_cascade = cv2.CascadeClassifier('./haar/haarcascades/haarcascade_eye.xml')
        faces = self.detectFaces(image_name)
    
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = []
        for (x1,y1,x2,y2) in faces:
            roi_gray = gray[y1:y2, x1:x2]
            eyes = eye_cascade.detectMultiScale(roi_gray,1.3,2)
            for (ex,ey,ew,eh) in eyes:
                result.append((x1+ex,y1+ey,x1+ex+ew,y1+ey+eh))
        return result
    
    
    #在原图像上框出眼睛.
    def drawEyes(self,image_name):
        eyes = self.detectEyes(image_name)
        if eyes:
            img = Image.open(image_name)
            draw_instance = ImageDraw.Draw(img)
            for (x1,y1,x2,y2) in eyes:
                draw_instance.rectangle((x1,y1,x2,y2), outline=(0, 0,255))
            img.save(image_name)
    
 
    #检测笑脸
    def detectSmiles(self,image_name):
        img = cv2.imread(image_name)
        smiles_cascade = cv2.CascadeClassifier("./haar/haarcascades/haarcascade_smile.xml")
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    
        smiles = smiles_cascade.detectMultiScale(gray,4,5)
        result = []
        for (x,y,width,height) in smiles:
            result.append((x,y,x+width,y+height))
        return result
    
    
    #在原图像上框出笑脸
    def drawSmiles(self,image_name):
        smiles = self.detectSmiles(image_name)
        if smiles:
            img = Image.open(image_name)
            draw_instance = ImageDraw.Draw(img)
            for (x1,y1,x2,y2) in smiles:
                draw_instance.rectangle((x1,y1,x2,y2), outline=(100, 100,0))
            img.save(image_name)

    def gender_classify(self,image_name,area):
        img = cv2.imread(image_name)
        img = img[area[1]:area[3],area[0]:area[2]]
        img = cv2.resize(img, (92,112),interpolation = cv2.INTER_AREA).flatten()

        ball=np.array([img])
        #
        np.set_printoptions(suppress=True)

        #取一张图片
        input_image = ball#
        #input_image = train_data.images[0:1]
        labels = train_data.labels[0:1]
        fig2,ax2 = plt.subplots(figsize=(2,2))
        #input_image=cv2.resize(input_image,( 92,112),interpolation=cv2.INTER_CUBIC)
        ax2.imshow(np.reshape(input_image, (112, 92,3)))
        #plt.show()

        sess = tf.Session()
        graph_path=os.path.abspath('./model/my-gender-v1.0.meta')
        model=os.path.abspath('./model/')

        server = tf.train.import_meta_graph(graph_path)
        server.restore(sess,tf.train.latest_checkpoint(model))

        graph = tf.get_default_graph()

        #填充feed_dict
        x = graph.get_tensor_by_name('input_images:0')
        y = graph.get_tensor_by_name('input_labels:0')
        feed_dict={x:input_image,y:labels}


        #第一层卷积+池化
        relu_1 = graph.get_tensor_by_name('relu_1:0')
        max_pool_1 = graph.get_tensor_by_name('max_pool_1:0')

        #第二层卷积+池化
        relu_2 = graph.get_tensor_by_name('relu_2:0')
        max_pool_2 = graph.get_tensor_by_name('max_pool_2:0')

        #第三层卷积+池化
        relu_3 = graph.get_tensor_by_name('relu_3:0')
        max_pool_3 = graph.get_tensor_by_name('max_pool_3:0')

        #全连接最后一层输出
        f_softmax = graph.get_tensor_by_name('f_softmax:0')


        #relu_1_r,max_pool_1_,relu_2,max_pool_2,relu_3,max_pool_3,f_softmax=sess.run([relu_1,max_pool_1,relu_2,max_pool_2,relu_3,max_pool_3,f_softmax],feed_dict)



        #----------------------------------各个层特征可视化-------------------------------




        #conv1 特征
        r1_relu = sess.run(relu_1,feed_dict)
        r1_tranpose = sess.run(tf.transpose(r1_relu,[3,0,1,2]))
        fig,ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
        for i in range(16):
            ax[i].imshow(r1_tranpose[i][0])
        plt.title('Conv1 16*112*92')
        #plt.show()

        #pool1特征
        max_pool_1 = sess.run(max_pool_1,feed_dict)
        r1_tranpose = sess.run(tf.transpose(max_pool_1,[3,0,1,2]))
        fig,ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
        for i in range(16):
            ax[i].imshow(r1_tranpose[i][0])
        plt.title('Pool1 16*56*46')
        #plt.show()


        #conv2 特征
        r2_relu = sess.run(relu_2,feed_dict)
        r2_tranpose = sess.run(tf.transpose(r2_relu,[3,0,1,2]))
        fig,ax = plt.subplots(nrows=1,ncols=32,figsize=(32,1))
        for i in range(32):
            ax[i].imshow(r2_tranpose[i][0])
        plt.title('Conv2 32*56*46')
        #plt.show()

        #pool2 特征
        max_pool_2 = sess.run(max_pool_2,feed_dict)
        tranpose = sess.run(tf.transpose(max_pool_2,[3,0,1,2]))
        fig,ax = plt.subplots(nrows=1,ncols=32,figsize=(32,1))
        for i in range(32):
            ax[i].imshow(tranpose[i][0])
        plt.title('Pool2 32*28*23')
        #plt.show()


        #conv3 特征
        r3_relu = sess.run(relu_3,feed_dict)
        tranpose = sess.run(tf.transpose(r3_relu,[3,0,1,2]))
        fig,ax = plt.subplots(nrows=1,ncols=64,figsize=(32,1))
        for i in range(64):
            ax[i].imshow(tranpose[i][0])
        plt.title('Conv3 64*28*23')
        #plt.show()

        #pool3 特征
        max_pool_3 = sess.run(max_pool_3,feed_dict)
        tranpose = sess.run(tf.transpose(max_pool_3,[3,0,1,2]))
        fig,ax = plt.subplots(nrows=1,ncols=64,figsize=(32,1))
        for i in range(64):
            ax[i].imshow(tranpose[i][0])
        plt.title('Pool3 64*14*12')
        #plt.show()

        result=sess.run(f_softmax,feed_dict)
        print(result)
        print(labels_text[np.argmax(result)])
 
if __name__ == '__main__':
    time1=datetime.now()
    detect_obj=DetectFaces()
    result=detect_obj.detectFaces('heat.jpg')
    
    time2=datetime.now()
    print("耗时："+str(time2-time1))
    if len(result)>0:
        print("有人存在！！---》人数为："+str(len(result)))
    else:
        print('视频图像中无人！！')
    for res in result:
        detect_obj.gender_classify('heat.jpg',res)
 
    #detect_obj.drawFaces('./resources/pic/slx.jpg')
    #detect_obj.drawSmiles('./resources/pic/slx.jpg')

    #detect_obj.saveFaces('./resources/pic/slx.jpg')




"""
上面的代码将眼睛、人脸、笑脸在不同的图像上框出，如果需要在同一张图像上框出，改一下代码就可以了。
总之，利用opencv里训练好的haar特征的xml文件，在图片上检测出人脸的坐标，利用这个坐标，我们可以将人脸区域剪切保存，也可以在原图上将人脸框出。剪切保存人脸以及用矩形工具框出人脸，本程序使用的是PIL里的Image、ImageDraw模块。
此外，opencv里面也有画矩形的模块，同样可以用来框出人脸。
"""
