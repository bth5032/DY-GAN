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

import matplotlib.pyplot as plt


import ROOT as r

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

        model.add(Dense(64, input_shape=output_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(8, activation='tanh'))
        model.add(Dense(8))

        model.summary()

        noise = Input(shape=output_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(64,input_shape=output_shape))
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
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=output_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=100):

        # # Load the dataset
        # # (X_train, _), (_, _) = mnist.load_data()
        # # X_train = np.random.random((10000,8)).astype(np.float32)
        # N = 100000
        # col1 = 0.02*np.random.random(N)+1
        # col2 = 0.02*np.random.random(N)+2
        # col3 = 0.02*np.random.random(N)+3
        # col4 = 0.02*np.random.random(N)+4
        # col5 = 0.02*np.random.random(N)+5
        # col6 = 0.02*np.random.random(N)+6
        # col7 = 0.02*np.random.random(N)+7
        # col8 = 0.02*np.random.random(N)+8
        # X_train = np.c_[col1,col2,col3,col4,col5,col6,col7,col8].astype(np.float32)

        # # X_train = np.loadtxt(open("coolio.txt", "r"), delimiter=",", skiprows=1)
        # X_train = np.loadtxt(open("data_cartesian.csv", "r"), delimiter=",", skiprows=1)
        # print X_train
        # print X_train.shape

        data = np.loadtxt(open("data_cartesian.csv", "r"), delimiter=",", skiprows=1)
        X_train = data[:,range(1,1+8)]
        invmass_data = data[:,0]

        # v1 = r.TLorentzVector()
        # v2 = r.TLorentzVector()
        # for img in X_train[np.random.randint(0, X_train.shape[0], 250)]:
        #     v1.SetPtEtaPhiE(img[1],img[2],img[3],img[0])
        #     v2.SetPtEtaPhiE(img[5],img[6],img[7],img[4])
        #     print (v1+v2).M(), img
        # # print gen_imgs

        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 8))

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
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        noise = np.random.normal(0, 1, (1,8))
        gen_imgs = self.generator.predict(noise)
        v1 = r.TLorentzVector()
        v2 = r.TLorentzVector()
        img = gen_imgs[0]
        v1.SetPtEtaPhiE(img[1],img[2],img[3],img[0])
        v2.SetPtEtaPhiE(img[5],img[6],img[7],img[4])
        print (v1+v2).M(), img
        # print gen_imgs

        # print(gen_imgs)


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=1024, save_interval=50)

    # # data = np.loadtxt(open("data.csv", "r"), delimiter=",", skiprows=1)
    # data = np.loadtxt(open("data_cartesian.csv", "r"), delimiter=",", skiprows=1)
    # x_data = data[:,range(1,1+8)]
    # y_data = data[:,0]
    # print x_data.shape
    # print y_data.shape

    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    # batch_size = 512
    # epochs = 9
    # save_to = "model.h5"
    # load_from = "model.h5"

    # if load_from and os.path.exists(load_from):
    #     model = load_model(load_from)
    #     model.summary()
    # else:
    #     model = Sequential()

    #     model.add(Dense(512, input_shape=output_shape,activation="tanh"))
    #     model.add(Dense(1))

    #     model.compile(loss='mean_squared_error',
    #                   optimizer='adam',
    #                   metrics=['accuracy'])
    #     model.summary()
        
    #     model.fit(x_train, y_train,
    #               batch_size=batch_size,
    #               nb_epoch=epochs,
    #               verbose=1,
    #               callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
    #               validation_data=(x_test, y_test))
    #     if save_to:
    #         model.save(save_to)



    # # score = model.evaluate(x_test, y_test, batch_size=batch_size)
    # print model.predict(x_test)
    # print y_test
    # # print score

