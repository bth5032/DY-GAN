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
from sklearn.preprocessing import *


from test.scaler import SymLogScaler

import pickle
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
input_shape = (8,)

d_epochinfo = {}
# ss = MinMaxScaler(feature_range=(-0.5, 0.5))
# ss = RobustScaler()
# ss = SymLogScaler()
# ss = StandardScaler()

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
        z = Input(shape=input_shape)
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

        model.add(Dense(64, input_shape=input_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(8))

        model.summary()

        noise = Input(shape=input_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256,input_shape=output_shape))
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

    def train(self, epochs, batch_size=128, tag=""):

        # data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)

        data = np.load("data_xyz.npy")
        X_train = data[:,range(1,1+8)]
        invmass_data = data[:,0]

        # NOTE. StandardScaler should be fit on training set
        # and applied the same to train and test, otherwise we 
        # introduce a bias
        # ss.fit(X_train)

        # import pickle
        # scalerfile = 'scaler.sav'
        # pickle.dump(ss, open(scalerfile, 'wb'))

        # print X_train
        # for i in range(8):
        #     print X_train[:,i].min(), X_train[:,i].max()
        # X_train = ss.transform(X_train).astype(np.float32)
        # print X_train
        # print X_train[:,0].min()
        # print X_train[:,0].max()
        # # X_test = ss.transform(X_test).astype(np.float32)

        # data = np.load("data_delphes.npa")
        # X_train = np.c_[
        #         data["lep1_e"],
        #         data["lep1_px"],
        #         data["lep1_py"],
        #         data["lep1_pz"],
        #         data["lep2_e"],
        #         data["lep2_px"],
        #         data["lep2_py"],
        #         data["lep2_pz"],
        #         ]
        # invmass_data = data["mll"]

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise_half = np.random.normal(0, 1, (half_batch, input_shape[0]))
            noise_full = np.random.normal(0, 1, (batch_size, input_shape[0]))

            # noise = np.random.normal(1,0.003, (half_batch,8))*imgs

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise_half)

            # Train the discriminator

            ones = np.ones((half_batch, 1))
            zeros = np.zeros((half_batch, 1))

            frac = 0.3*np.exp(-epoch/3000.)
            if frac > 0.005:
                # print np.random.randint(0, len(ones), int(frac*len(ones)))
                ones[np.random.randint(0, len(ones), int(frac*len(ones)))] = 0
                zeros[np.random.randint(0, len(zeros), int(frac*len(zeros)))] = 1

            d_loss_real = self.discriminator.train_on_batch(imgs, ones)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, zeros)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------


            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise_full, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % 20 == 0:

                noise = np.random.normal(0, 1, (5000,input_shape[0]))
                gen_imgs = self.generator.predict(noise)
                # gen_imgs = ss.inverse_transform(gen_imgs)
                masses=M(gen_imgs[:,0],gen_imgs[:,1],gen_imgs[:,2],gen_imgs[:,3],gen_imgs[:,4],gen_imgs[:,5],gen_imgs[:,6],gen_imgs[:,7])
                masses = masses[np.isfinite(masses)]
                print masses.mean(), masses.std()

                if "epoch" not in d_epochinfo: 
                    d_epochinfo["epoch"] = []
                    d_epochinfo["d_acc"] = []
                    d_epochinfo["d_loss"] = []
                    d_epochinfo["g_loss"] = []
                    d_epochinfo["mass_mu"] = []
                    d_epochinfo["mass_sig"] = []
                else:
                    d_epochinfo["epoch"].append(epoch)
                    d_epochinfo["d_acc"].append(100*d_loss[1])
                    d_epochinfo["d_loss"].append(d_loss[0])
                    d_epochinfo["g_loss"].append(g_loss)
                    d_epochinfo["mass_mu"].append(masses.mean())
                    d_epochinfo["mass_sig"].append(masses.std())

                np.save("progress/{}/pred_{}.npy".format(tag,epoch), gen_imgs)
                pickle.dump(d_epochinfo, open("progress/{}/history.pkl".format(tag),'w'))

            if epoch % 10000 == 0: 
                noise = np.random.normal(0, 1, (10000,input_shape[0]))
                gen_imgs = self.generator.predict(noise)
                masses=M(gen_imgs[:,0],gen_imgs[:,1],gen_imgs[:,2],gen_imgs[:,3],gen_imgs[:,4],gen_imgs[:,5],gen_imgs[:,6],gen_imgs[:,7])
                print("mean: %s, std: %s " % (masses.mean(), masses.std()))
                self.generator.save("progress/{}/gen_{}.weights".format(tag,epoch))


if __name__ == '__main__':
    # tag = "v1noise2"
    # tag = "v1noise1"
    tag = "vdecaynoise"
    os.system("mkdir -p progress/{}/".format(tag))

    gan = GAN()
    # gan.train(epochs=100002, batch_size=10000, tag=tag)
    gan.train(epochs=100002, batch_size=5000, tag=tag)
