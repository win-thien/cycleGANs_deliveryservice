'''
THIEN WIN

BrainStation Data Science Capstone
April 2022
'''

'''
[A] Function Definition

Introduction

This python file is a collection of functions used throughout the project to help build the cycleGAN model. 
In subsequent notebooks, this file and containing functions willbe imported for use. The aim of this file 
is to create a cohesive document where quick changes to functions can be achieved and provide a cleaner 
notebook downstream.
'''

#import required dependencies and libraries
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import pathlib
import matplotlib.pyplot as plt

#HELPER FUNCTIONS

def normalize(image):
    '''
    Helper function for decoding image. 
    '''
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def center_crop(image):
    '''
    Helper function to produce square image first.
    i.e. SG photos are 1038x1920 ---> crops from img center ---> 1038x1038
    '''
    offset_height = 0
    offset_width = (tf.shape(image)[1] - tf.shape(image)[0]) // 2
    target_height = tf.shape(image)[0]
    target_width = tf.shape(image)[0]
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)


def resize(image, size=[256,256]):
    '''
    Helper function to resize image to target value i.e. 256x256
    '''
    return tf.image.resize(image, size, preserve_aspect_ratio=True, method='bilinear')


#PREPROCESSING FUNCTIONS
'''
Preprocessing for test and train data were defined separately for ease of future modifcation.
For future datasets, not all helper functions may be required.
'''

def preprocess_test_SG(image):
    '''
    Combination of helper functions to preprocess test images for Domain A (Studio Ghibli).
    '''
    image = normalize(image)
    image = center_crop(image)
    image = resize(image)
    return image


def preprocess_train_SG(image):
    '''
    Combination of helper functions to preprocess train images for Domain A (Studio Ghibli).
    '''
    image = normalize(image)
    image = center_crop(image)
    image = resize(image)
    return image


def preprocess_test_photo(image):
    '''
    Combination of helper functions to preprocess test images for Domain B (photos).
    '''
    image = normalize(image)
    return image


def preprocess_train_photo(image):
    '''
    Combination of helper functions to preprocess train images for Domain B (photos).
    '''
    image = normalize(image)
    return image


#MODEL FUNCTIONS
'''
Model functions include downsample, upsample, discriminator and generator function definitions.
'''

def downsample(filters, size, apply_instancenorm=True):
    '''
    The downsample (decoder) takes the number of filters (nodes), kernel size and whether or not to use 
    instance normalization as an argument as discussed in the original paper. Instance normalization is
    is set to True as default. Refer to written report for source. This is used in the discriminator and generator architecture. 
    '''
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    '''
    The upsample (encoder) takes the number of filters (nodes), kernel size and whether or not to apply 
    a dropout which is set to False by default as an argument as discussed in the original paper. Refer to written
    report for source. This is used in the discriminator and generator architecture. 
    '''
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def Discriminator():
    '''
    The discriminator is as described the original paper and uses a 70x70 PatchGAN classifier
    and is the same used for the Pix2Pix method. The intuitive aspect of the discriminator is 
    that it outputs a "patch" of the image and tries to classify it as real or fake.
    https://www.tensorflow.org/tutorials/generative/pix2pix
    '''
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=inp, outputs=last)


def Generator():
    '''
    The generator differs from what was proposed in the paper. The paper calls for a ResNET based
    generator where as the one defined in this function is a UNet based one. 

    For future iterations, a second generator type might be define to based on a ResNet type.
    '''
    
    OUTPUT_CHANNELS = 3

    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), 
        downsample(128, 4), 
        downsample(256, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
        downsample(512, 4), 
        downsample(512, 4) 
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4), 
        upsample(256, 4), 
        upsample(128, 4), 
        upsample(64, 4) 
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') 

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)



