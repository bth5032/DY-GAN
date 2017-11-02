# from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

import sys
sys.path.append("../")

import plottery.plottery as ply
import plottery.utils as plu
import matplotlib.pyplot as plt

import ROOT as r
from physicsfuncs import M

import sys
import os

import numpy as np

output_shape = (8,)
class GAN():
    def __init__(self):
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1

        # optimizer = Adam(0.0002, 0.5)
        optimizer = "adadelta"

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=output_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(32, input_shape=output_shape))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(8))

        model.summary()

        noise = Input(shape=output_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256,input_shape=output_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()

        img = Input(shape=output_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128):

        # data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)
        data = np.load("data_xyz.npy")
        X_train = data[:,range(1,1+8)]
        invmass_data = data[:,0]

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 8))

            # noise = np.random.normal(1,0.003, (half_batch,8))*imgs

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 8))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [tot loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, g_loss+d_loss[0]))

            # If at save interval => save generated image samples
            self.save_imgs(epoch)

    def save_imgs(self, epoch):
        if epoch % 20 == 0:
            noise = np.random.normal(0, 1, (1000,8))
            gen_imgs = self.generator.predict(noise)
            print gen_imgs
            masses=M(gen_imgs[:,0],gen_imgs[:,1],gen_imgs[:,2],gen_imgs[:,3],gen_imgs[:,4],gen_imgs[:,5],gen_imgs[:,6],gen_imgs[:,7])
            masses = masses[np.isfinite(masses)]
            np.save("progress/pred_{}.npy".format(epoch), gen_imgs)
            print masses.mean(), masses.std()
            # h1 = r.TH1F("h1","masses",50,0,500)
            # plu.fill_fast(h1, masses)
            # if h1.Integral()>0.1:
            #     h1.Scale(1./h1.Integral())
            # ply.plot_hist(
            #     bgs=[h1],
            #     legend_labels = ["pred"],
            #     options = {
            #       "do_stack": False,
            #       "yaxis_log": False,
            #       "output_name": "masses.pdf",
            #       "output_ic": True,
            #       }
            #     )
        if epoch % 10000 == 0: 
            noise = np.random.normal(0, 1, (10000,8))
            gen_imgs = self.generator.predict(noise)
            masses=M(gen_imgs[:,0],gen_imgs[:,1],gen_imgs[:,2],gen_imgs[:,3],gen_imgs[:,4],gen_imgs[:,5],gen_imgs[:,6],gen_imgs[:,7])
            print("mean: %s, std: %s " % (masses.mean(), masses.std()))
            self.generator.save("gen_%i.weights" % epoch)


if __name__ == '__main__':
    os.system("mkdir progress")

    gan = GAN()
    gan.train(epochs=100002, batch_size=10000)
