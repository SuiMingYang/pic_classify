import tensorflow as tf
import numpy as np
import os
from cv2 import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./product-recognition/data/",one_hot=True)

# # 解决中文不显示问题
# from matplotlib.font_manager import _rebuild
# _rebuild() #reload一下

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names_zh = [u'短袖', u'裤子', u'套衫', u'裙子', u'大衣',
               u'凉鞋', u'衬衫', u'运动鞋', u'包', u'短靴']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label1, img = predictions_array, np.argmax(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label1:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names_zh[predicted_label],
                                100*np.max(predictions_array),
                                class_names_zh[true_label1]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, np.argmax(true_label[i])
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

print(mnist.train.images.shape)
#(55000, 784)
(train_image,train_label),(test_image,test_label)=(mnist.train.images,mnist.train.labels),(mnist.test.images,mnist.test.labels)
train_image=train_image.reshape(mnist.train.images.shape[0],28,28)
test_image=test_image.reshape(mnist.test.images.shape[0],28,28)
alrothy=tf.keras.Sequential()
alrothy.add(tf.keras.layers.Flatten(input_shape=(28,28)))
alrothy.add(tf.keras.layers.Dense(128,activation='relu'))
alrothy.add(tf.keras.layers.Dropout(0.5))
alrothy.add(tf.keras.layers.Dense(64,activation='relu'))
alrothy.add(tf.keras.layers.Dense(10,activation='softmax'))
alrothy.compile(optimizer=tf.keras.optimizers.Adamax(lr=0.01),loss='categorical_crossentropy',metrics=['acc'])

alrothy.fit(train_image,train_label,epochs=1)
test_loss, test_acc=alrothy.evaluate(test_image,test_label,verbose=2)

url='./product-recognition/pic/'
valid=[]
for i in os.listdir(url):
    for j in os.listdir(url+i+'/'):
        if j!='.DS_Store':
            pic1=cv2.imread(url+i+'/'+j)
            pic1 = cv2.resize(pic1,(28,28),interpolation=cv2.INTER_CUBIC)
            #cv2.imshow("img",img_resize)
            pic1=cv2.cvtColor(pic1,cv2.COLOR_RGB2GRAY)

            valid.append(np.array(pic1))

pred=alrothy.predict(np.array(valid[0:15])/255)
# pred=alrothy.predict(test_image/255)
alrothy.save('product_classify.h5')
model = tf.keras.models.load_model('product_classify.h5')
# print(class_names[np.argmax(pred[0])])

# 测试数据
pred=alrothy.predict(test_image)
print(np.argmax(pred[0]))
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, pred[i], test_label, test_image)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pred[i], test_label)
plt.tight_layout()
plt.show()

# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, pred[i], [[0,0,0,0,0,0,0,0,0,1]]*15, valid[0:15])
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, pred[i], [[0,0,0,0,0,0,0,0,0,1]]*15)
# plt.tight_layout()
# plt.show()








