import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#import all functions from created file
from cycleGAN_functions import *

import tensorflow_addons as tfa


strategy = tf.distribute.get_strategy()

with strategy.scope():
    SG_generator = Generator() # transforms photos to SG-type images
    photo_generator = Generator() # transforms SG images to be more like photos

    SG_discriminator = Discriminator() # differentiates real SG images and generated SG images
    photo_discriminator = Discriminator() # differentiates real photos and generated photos


#define discriminator loss function
with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5


#define generator loss function
with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)


#define cycle consistency loss function
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return LAMBDA * loss1


#define identity consistency loss function
with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss


with strategy.scope():
    SG_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    SG_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


class CycleGan(keras.Model):
    def __init__(
        self,
        SG_generator,
        photo_generator,
        SG_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__() #temporary object that allows us to access methods of the base class
        self.SG_gen = SG_generator
        self.p_gen = photo_generator
        self.SG_disc = SG_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        SG_gen_optimizer,
        p_gen_optimizer,
        SG_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile() #temporary object that allows us to access methods of the base class
        self.SG_gen_optimizer = SG_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.SG_disc_optimizer = SG_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_SG, real_photo = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # photo to SG back to photo
            fake_SG = self.SG_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_SG, training=True)

            # SG to photo back to SG
            fake_photo = self.p_gen(real_SG, training=True)
            cycled_SG = self.SG_gen(fake_photo, training=True)

            # generating itself
            same_SG = self.SG_gen(real_SG, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_SG = self.SG_disc(real_SG, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_SG = self.SG_disc(fake_SG, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            SG_gen_loss = self.gen_loss_fn(disc_fake_SG)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_SG, cycled_SG, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_SG_gen_loss = SG_gen_loss + total_cycle_loss + self.identity_loss_fn(real_SG, same_SG, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            SG_disc_loss = self.disc_loss_fn(disc_real_SG, disc_fake_SG)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        SG_generator_gradients = tape.gradient(total_SG_gen_loss,
                                                  self.SG_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        SG_discriminator_gradients = tape.gradient(SG_disc_loss,
                                                      self.SG_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.SG_gen_optimizer.apply_gradients(zip(SG_generator_gradients,
                                                 self.SG_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.SG_disc_optimizer.apply_gradients(zip(SG_discriminator_gradients,
                                                  self.SG_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        
        return {
            "SG_gen_loss": total_SG_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "SG_disc_loss": SG_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }
