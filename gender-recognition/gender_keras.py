from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import json
import os

from tensorflow.keras.applications.resnet50 import ResNet50

batch_size = 40
epochs = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224

num_classes=2
image_input=180


PATH = os.path.join('./gender-recognition/data/')


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')


train_male_dir = os.path.join(train_dir, 'male')
train_female_dir = os.path.join(train_dir, 'female')

validation_male_dir = os.path.join(train_dir, 'male')
validation_female_dir = os.path.join(train_dir, 'female')






num_male_tr = len(os.listdir(train_male_dir))
num_female_tr = len(os.listdir(train_female_dir))




num_male_val = len(os.listdir(validation_male_dir))
num_female_val = len(os.listdir(validation_female_dir))




total_train = num_male_tr+num_female_tr
total_val = num_male_val + num_female_val






print("Total training images:", total_train)
print("Total validation images:", total_val)




# 训练集
# 对训练图像应用了重新缩放，45度旋转，宽度偏移，高度偏移，水平翻转和缩放增强。
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    width_shift_range=0.1,
                    height_shift_range=0.1
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

# 验证集

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')




# 创建模型


model=ResNet50(include_top=True, weights=None,classes=num_classes)
# 编译模型

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# 模型总结
model.summary()


# 模型保存格式定义

model_class_dir='./gender-recognition/model/'
class_indices = train_data_gen.class_indices
class_json = {}
for eachClass in class_indices:
    class_json[str(class_indices[eachClass])] = eachClass

with open(os.path.join(model_class_dir, "model_class.json"), "w+") as json_file:
    json.dump(class_json, json_file, indent=4, separators=(",", " : "),ensure_ascii=True)
    json_file.close()
print("JSON Mapping for the model classes saved to ", os.path.join(model_class_dir, "model_class.json"))



model_name = 'model_ex.h5'

trained_model_dir=model_class_dir
model_path = os.path.join(trained_model_dir, model_name)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_acc',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            mode='max',
            period=1)


def lr_schedule(epoch):
    # Learning Rate Schedule

    lr =1e-3
    total_epochs =epoch

    check_1 = int(total_epochs * 0.9)
    check_2 = int(total_epochs * 0.8)
    check_3 = int(total_epochs * 0.6)
    check_4 = int(total_epochs * 0.4)

    if epoch > check_1:
        lr *= 1e-4
    elif epoch > check_2:
        lr *= 1e-3
    elif epoch > check_3:
        lr *= 1e-2
    elif epoch > check_4:
        lr *= 1e-1

    return lr



#lr_scheduler =tf.keras.callbacks.LearningRateScheduler(lr_schedule)


lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)


num_train = len(train_data_gen.filenames)
num_test = len(val_data_gen.filenames)

print(num_train,num_test)

# 模型训练
# 使用fit_generator方法ImageDataGenerator来训练网络。

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(num_train / batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(num_test / batch_size),
    callbacks=[checkpoint,lr_scheduler])




