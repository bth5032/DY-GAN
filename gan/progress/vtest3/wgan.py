# from __future__ import print_function

"""
Tried to follow
https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html
And actually after following that, I saw that my implementation actually matched
https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
(this is the guy we took the original GAN from)
so that was reassuring.
"""

import os
import tensorflow as tf
tf.set_random_seed(42)
# if "USE_ONE_CPU" in os.environ or True:
if "USE_ONE_CPU" in os.environ:
    print ">>> using two CPUs"
    config = tf.ConfigProto(
          intra_op_parallelism_threads=2,
          inter_op_parallelism_threads=2) # if this one is 2, then stable?
else:
    print ">>> possibly using many CPUs"
    config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras.backend.tensorflow_backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from sklearn.preprocessing import *
import numpy as np
np.random.seed(42)

import argparse
import time
import pickle
import sys
import os
sys.path.append("../")

# import watch
from physicsfuncs import Minv, get_metrics, cartesian_to_ptetaphi

# from pyfiles.tqdm import format_meter, StatusPrinter

def wgan_d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class GAN():
    def __init__(self, args):

        self.args = args

        self.tag = "vtest3"
        self.input_file = "data_xyz.npy"
        self.batch_size = 500
        self.nepochs_max = 50000
        self.nepochs_dump_pred_metrics = 250
        self.nepochs_dump_history = 500
        self.output_shape = (8,)
        self.noise_shape = (8,)
        # self.scaler = RobustScaler()
        self.scaler = None

        os.system("mkdir -p progress/{}/".format(self.tag))
        os.system("cp wgan.py progress/{}/".format(self.tag))

        self.data_ref = None
        self.d_epochinfo = {}

        optimizer_d = RMSprop(lr=0.001)
        optimizer_g = RMSprop(lr=0.01)
        # optimizer_d = "RMSprop"
        # optimizer_g = "RMSprop"

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=wgan_d_loss, optimizer=optimizer_d, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=wgan_d_loss, optimizer=optimizer_g)

        # The generator takes noise as input and generated imgs
        z = Input(shape=self.noise_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss=wgan_d_loss, optimizer=optimizer_g)

    def build_generator(self):

        model = Sequential()

        ## Head
        model.add(Dense(64, input_shape=self.output_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.output_shape[0]))

        model.summary()

        noise = Input(shape=self.noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        weight_init = RandomNormal(mean=0.2, stddev=0.02)

        model.add(Dense(128,input_shape=self.output_shape,kernel_initializer=weight_init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dense(128,kernel_initializer=weight_init))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(512,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1024,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1024,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(512,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(256,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(128,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(64,kernel_initializer=weight_init))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        ## Tail
        # WGAN has no squashing of the output of D. Thus, linear 
        # activation here
        model.add(Dense(1,activation='linear'))
        model.summary()

        img = Input(shape=self.output_shape)
        validity = model(img)

        return Model(img, validity)

    def clip_d_weights(self):

        # Tried both and same result

        weights = [np.clip(w, -0.5, 0.5) for w in self.discriminator.get_weights()]
        self.discriminator.set_weights(weights)

        # for l in self.discriminator.layers:
        #     weights = l.get_weights()
        #     weights = [np.clip(w, -0.01,0.01) for w in weights]
        #     l.set_weights(weights)

    def train(self):

        data = np.load(self.input_file)
        X_train = data[:,range(1,1+8)]
        self.data_ref = X_train[:500]

        if self.scaler:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train).astype(np.float32)
            pickle.dump(self.scaler, open("progress/{}/scaler.pkl".format(self.tag),'w'))

        half_batch = int(self.batch_size / 2)

        prev_weights = None
        #sp = StatusPrinter(sys.stderr)
        tstart = time.time()
        extra = ""
        for epoch in range(self.nepochs_max):
            

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # WGAN calls D a "critic" because it 
            # doesn't squash the output -- it "critiques" the quality of
            # the input event by giving an unbounded number
            # WGAN also trains D `ncritic` times per epoch,
            # and sometimes a lot more, depending on the epoch number
            # Want to always keep D ahead of G
            ncritic = 5
            if (epoch % 1000) < 15 or epoch % 500 == 0: 
                # first 25 epochs every 1000 epochs, and every 500th epoch, we train D 50 times instead of 5
                ncritic = 20
            for icritic in range(ncritic):

                # Select a random half batch of images and make some noise as input
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]
                noise_half = np.random.normal(0, 1, (half_batch, self.noise_shape[0]))


                # Generate a half batch of new images
                gen_imgs = self.generator.predict(noise_half)
                # Use these ones as truth values
                # Note, the loss function is the product of prediction and truth values
                # So predicting 0.7 for a real sample gives loss of 0.7. We want to maximize this
                # number, but Keras minimizes loss. So the truth value for a real sample is actually
                # NEGATIVE one, hence the `-ones` in the d_loss_real
                # And for fake samples, we want the opposite (`ones` intead of `-ones`)
                ones = np.ones((half_batch, 1))
                d_loss_real = self.discriminator.train_on_batch(imgs, -ones)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, ones)
                # The total loss is therefore the sum of the two (so lower is better)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Clip the discriminator weights
                self.clip_d_weights()

            # if not prev_weights: prev_weights = self.discriminator.get_weights()
            # print "START DWEIGHTS >>>>>"
            # # print np.array(prev_weights)-np.array(self.discriminator.get_weights())
            # for l in self.discriminator.layers:
            #     sum_weights = [np.sum(np.array(x)) for x in l.get_weights()]
            #     print l,sum_weights
            # print "END DWEIGHTS <<<<<"
            # if epoch == 15:
            #     print prev_weights - self.discriminator.get_weights()
            #     print self.discriminator.predict(imgs)[:10]
            #     print self.discriminator.predict(gen_imgs)[:10]
            #     break

            # ---------------------
            #  Train Generator
            # ---------------------

            # The generator wants the discriminator to label the generated samples as valid
            valid_y = np.array([1] * self.batch_size)

            noise_full = np.random.normal(0, 1, (self.batch_size, self.noise_shape[0]))

            # Train the generator
            # Again, since the loss function we use is maximum when real samples get high predictions, we want to invert
            # valid_y to allow Keras to improve the generator by minimizing the loss
            g_loss = self.combined.train_on_batch(noise_full, -valid_y)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1.-d_loss[0], 1.-g_loss))
            # extra = "[D loss: %f] [G loss: %f]" % (1.-d_loss[0], 1.-g_loss)
            # sp.print_status(format_meter(epoch, self.nepochs_max, time.time()-tstart, size=25,extra=extra))

            if epoch % self.nepochs_dump_pred_metrics in (self.nepochs_dump_pred_metrics-2, self.nepochs_dump_pred_metrics-1, 0, 1, 2) and epoch > 3:

                noise_test = np.random.normal(0, 1, (500,self.noise_shape[0]))
                gen_imgs = self.generator.predict(noise_test)

                if self.scaler:
                    gen_imgs = self.scaler.inverse_transform(gen_imgs)

                masses = Minv(gen_imgs)
                try:
                    metric1, metric2 = get_metrics(gen_imgs, self.data_ref)
                except:
                    metric1, metric2 = 0., 0.
                masses = masses[np.isfinite(masses)]
                print masses.mean(), masses.std()

                if "epoch" not in self.d_epochinfo: 
                    self.d_epochinfo["epoch"] = []
                    self.d_epochinfo["d_acc"] = []
                    self.d_epochinfo["d_loss"] = []
                    self.d_epochinfo["g_loss"] = []
                    self.d_epochinfo["mass_mu"] = []
                    self.d_epochinfo["mass_sig"] = []
                    self.d_epochinfo["metric1"] = []
                    self.d_epochinfo["metric2"] = []
                    self.d_epochinfo["time"] = []
                    self.d_epochinfo["args"] = vars(self.args)
                else:
                    self.d_epochinfo["epoch"].append(epoch)
                    self.d_epochinfo["d_acc"].append(100*d_loss[1])
                    self.d_epochinfo["d_loss"].append(d_loss[0])
                    self.d_epochinfo["g_loss"].append(g_loss)
                    self.d_epochinfo["mass_mu"].append(masses.mean())
                    self.d_epochinfo["mass_sig"].append(masses.std())
                    self.d_epochinfo["metric1"].append(metric1)
                    self.d_epochinfo["time"].append(time.time())

                pickle.dump(self.d_epochinfo, open("progress/{}/history.pkl".format(self.tag),'w'))
                print("Avg Mass:", self.d_epochinfo["mass_mu"])
                self.generator.save("%s/gen_%i.weights" % ("progress/{}".format(self.tag), epoch))

            if epoch % self.nepochs_dump_history == 0 and epoch > 0:
                fname = "progress/{}/history.pkl".format(self.tag)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    
    args = parser.parse_args()

    gan = GAN(args)
    gan.train()
