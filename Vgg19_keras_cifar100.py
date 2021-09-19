# ========================================================== #
# File name: Vgg10_keras_cifar10.py
# Author: BIGBALLON
# Python Version: 3.6
<<<<<<< HEAD
# Tensorflow-gpu Version: 2.1.0
# Keras Version: 2.3.1
# dataset: cifar100
#Result: test accuracy about 72.56 ~ 73.57%
=======
# Tensorflow Version: 2.1.0
# Keras Version: 2.3.1
# dataset: cifar100
>>>>>>> f6729bc (change)
# ========================================================== #
import os
import sys
import datetime
import keras
import tensorflow
import numpy as np
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
# save the output results to an external file
def savefile(name):
    deskpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.'))
    fullpath = deskpath + name + '.txt'
    file = open(fullpath, 'w')
filename = 'ANN_cifar100'
savefile(filename)
output = sys.stdout
outputfile = open(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.')) + filename + '.txt',
                  'w')
sys.stdout = outputfile
time_start= datetime.datetime.now()
print("start     "+time_start.strftime('%Y.%m.%d-%H:%M:%S'))
num_classes  = 100
batch_size   = 256
epochs       = 300
iterations   = 391
dropout      = 0.5
weight_decay = 0.0001
log_filepath = os.path.join('vgg19_retrain_logs—2')

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    if epoch < 220:
        return 0.001
    return 0.0001

#WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
#filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
#filepath = '/home/mai/cifar-10-cnn/3_Vgg19_Network/vgg19_retrain_logs/retrain.h5'
# data loading
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# data preprocessing 
#x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
#x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
#x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
#x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
#x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
#x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

x_train = x_train / 255.0
x_test = x_test / 255.0



# build model
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
# model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
# model.add(Dropout(0.1))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
# model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
# model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool'))

# model modification for cifar-100
model.add(Flatten(name='flatten'))
model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa100'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
<<<<<<< HEAD
model.add(Dropout(0.6))
=======
model.add(Dropout(dropout))      
>>>>>>> f6729bc (change)
model.add(Dense(100, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa100'))
model.add(BatchNormalization())
#model.add(Activation('softmax'))

# load pretrained weight from VGG19 by name      
#model.load_weights(filepath, by_name=True)

# -------- optimizer setting -------- #
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)


print('Using real-time data augmentation.')
#datagen = ImageDataGenerator(horizontal_flip=True,
#        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

#datagen.fit(x_train)
datagen = ImageDataGenerator(
  #  featurewise_center=False,  # set input mean to 0 over the dataset
  #  samplewise_center=False,  # set each sample mean to 0
  #  featurewise_std_normalization=False,  # divide inputs by std of the dataset
  #  samplewise_std_normalization=False,  # divide each input by its std
  #  zca_whitening=False,  # apply ZCA whitening
 #   rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True  # randomly flip images
  #  vertical_flip=False
   )  # randomly flip images

datagen.fit(x_train)

checkpoint = ModelCheckpoint('best_model_improved-100.h5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
cbks = [change_lr,tb_cb,checkpoint]                                         # automatically depending on the quantity to monitor 
model._get_distribution_strategy = lambda: None
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
model.save('cifar100.h5')
time_end= datetime.datetime.now()
print("end     "+time_end.strftime('%Y.%m.%d-%H:%M:%S'))
outputfile.close()