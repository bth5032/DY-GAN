# from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
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
import physicsfuncs as pf

import sys, os, time, argparse

import numpy as np

output_shape = (14,)
input_shape = (2*output_shape[0],)
tag="test"

def Mass(x):
    return (x[0]+x[4])**2 - (x[1]+x[5])**2 - (x[2]+x[6])**2 - (x[3]+x[7])**2

class GAN():
    def __init__(self, output_dir):
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.output_dir=output_dir

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
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
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

        #model.add(Dense(128,activation="relu",input_shape=gen_output_shape))
        model.add(Lambda(lambda x: K.concatenate([x,Mass(x)]), input_shape=output_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(128,activation="relu"))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(128,activation="relu"))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(128,activation="relu"))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(128,activation="relu"))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(128,activation="relu"))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(64,activation="relu"))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(32,activation="relu"))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(16,activation="relu"))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(8,activation="relu"))
        model.add(Dense(8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()

        img = Input(shape=output_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=20):
        data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)
        X_train = pf.getCartRowsWithPtEtaPhiAppended(data[:,1,1+8])
        invmass_data = data[:,0]

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            rand_rows = np.random.randint(0, X_train.shape[0], half_batch)
            truth_bias = X_train[rand_rows]
            noise = np.random.normal(0, 1, (half_batch,input_shape[0]-truth_bias.shape[1]))
            gen_input=np.c_[noise, truth_bias]
            #noise = np.random.normal(0, 1, (half_batch, input_shape[0] - ))

            # noise = np.random.normal(1,0.003, (half_batch,8))*imgs

            # Generate a half batch of new images
            gen_output = self.generator.predict(gen_input)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(truth_bias, np.ones((half_batch, 1))) #use truth bias rows as real ones.
            d_loss_fake = self.discriminator.train_on_batch(gen_output, np.zeros((half_batch, 1))) #use gen output from truth bias rows as output.
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            rand_rows = np.random.randint(0, X_train.shape[0], batch_size)
            truth_bias = X_train[rand_rows]
            noise = np.random.normal(0, 1, (batch_size,input_shape[0]-truth_bias.shape[1]))
            gen_input=np.c_[noise, truth_bias]
            #noise = np.random.normal(0, 1, (batch_size, 8))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(gen_input, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [tot loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, g_loss+d_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                rand_rows = np.random.randint(0, X_train.shape[0], half_batch)
                truth_bias = X_train[rand_rows]
                noise = np.random.normal(0, 1, (half_batch,input_shape[0]-truth_bias.shape[1]))
                gen_input_extra=np.c_[noise, truth_bias]
                self.save_imgs(epoch, gen_input_extra)

    def save_imgs(self, epoch, gen_input):
        gen_output = self.generator.predict(gen_input)
        masses=pf.M(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
        print(masses.mean(), masses.std())
        np.save(output_dir+"/pred_%s.npy" % epoch, gen_output)
        if epoch % 2000 == 0: 
            self.generator.save("%s/gen_%i.weights" % (self.output_dir, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tag", "-t", help="Tag for the run", type=str)
    args = parser.parse_args()
    if args.tag:
        tag=args.tag

    #make a new model name based on the timestamp...
    output_dir="model/model_%d_%s/" % (int(time.time()), tag)
    os.mkdir(output_dir)
    #copy the GAN over so that we know what model was trained...
    os.system("cp gan.py %s" % output_dir)

    gan = GAN(output_dir)
    gan.train(epochs=100002, batch_size=10000, save_interval=20)

