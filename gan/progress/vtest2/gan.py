# from __future__ import print_function

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
from keras.optimizers import Adam
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


def add_invmass_from_8cartesian(x):
    invmass = K.sqrt(
                (x[:,0:1]+x[:,4:5])**2-
                (x[:,1:2]+x[:,5:6])**2-
                (x[:,2:3]+x[:,6:7])**2-
                (x[:,3:4]+x[:,7:8])**2
                )
    return K.concatenate([x,invmass])

def fix_outputs(x):
    """
    Take nominal delphes format of 21 columns and fix some columns
    """
    return K.concatenate([
        # x[:,0:21],
        x[:,0:8], # epxpypz for lep1,lep2
        K.sign(x[:,8:9]), # lep1 charge
        x[:,9:10], # lep1 iso
        K.sign(x[:,10:11]), # lep2 charge
        x[:,11:12], # lep2 iso
        K.round(x[:,12:13]), # nvtxs
        x[:,13:15], # met, metphi
        K.round(x[:,15:16]), # ngenjets
        x[:,16:21], # jet pts
        ])

class GAN():
    def __init__(self, args):

        self.args = args

        self.tag = args.tag
        self.input_file = str(args.input_file)
        self.noise_shape = (int(args.noise_size),)
        self.output_shape = (int(args.output_size),)
        self.noise_type = int(args.noise_type)
        self.ntest_samples = int(args.ntest_samples)
        self.nepochs_dump_pred_metrics = int(args.nepochs_dump_pred_metrics)
        self.nepochs_dump_history = int(args.nepochs_dump_history)
        self.nepochs_dump_models = int(args.nepochs_dump_models)
        self.nepochs_max = int(args.nepochs_max)
        self.batch_size = int(args.batch_size)
        self.do_concatenate_disc = args.do_concatenate_disc
        self.do_concatenate_gen = args.do_concatenate_gen
        self.do_batch_normalization_disc = args.do_batch_normalization_disc
        self.do_batch_normalization_gen = args.do_batch_normalization_gen
        self.do_soft_labels = args.do_soft_labels
        self.do_noisy_labels = args.do_noisy_labels
        self.do_tanh_gen = args.do_tanh_gen
        self.nepochs_decay_noisy_labels = int(args.nepochs_decay_noisy_labels)
        self.use_ptetaphi_additionally = args.use_ptetaphi_additionally
        self.optimizer_gen = args.optimizer_gen 
        self.optimizer_disc = args.optimizer_disc 
        self.depth_disc = args.depth_disc 
        self.width_disc = args.width_disc 
        self.depth_gen = args.depth_gen 
        self.width_gen = args.width_gen 
        self.beefy_generator = args.beefy_generator 
        self.add_invmass_disc = args.add_invmass_disc 
        self.fix_delphes_outputs = args.fix_delphes_outputs
        self.use_delphes = args.use_delphes

        os.system("mkdir -p progress/{}/".format(self.tag))
        os.system("cp gan.py progress/{}/".format(self.tag))

        self.scaler_type = args.scaler_type
        self.scaler = None
        if args.scaler_type.lower() == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1.,1.))
        elif args.scaler_type.lower() == "robust":
            self.scaler = RobustScaler()
        elif args.scaler_type.lower() == "standard":
            self.scaler = StandardScaler()

        self.data_ref = None
        self.d_epochinfo = {}

        # optimizer = Adam(0.0002, 0.5)
        optimizer_d = self.optimizer_disc
        # optimizer_d = "sgd"
        optimizer_g = self.optimizer_gen
        # optimizer_g = "adam"

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer_g)

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
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g)

    def build_generator(self):

        model = Sequential()

        ## Head
        model.add(Dense(64, input_shape=self.noise_shape))
        if self.do_batch_normalization_gen:
            model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        if self.do_concatenate_gen:
            model.add(Lambda(lambda x: K.concatenate([x*x,x])))
            model.add(LeakyReLU(alpha=0.2))
       
        ## Main Body
        if self.depth_gen > 0 and self.width_gen > 0:
            for level in xrange(0,self.depth_gen): 
                model.add(Dense(width_gen/(2**level))) #Triangle with width halved at each level
                model.add(LeakyReLU(alpha=0.2))
        elif self.beefy_generator:
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
        else:
            model.add(Dense(128))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(128))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(128))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(64))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(32))
            model.add(LeakyReLU(alpha=0.2))

        ## Tail
        model.add(Dense(self.output_shape[0]))
        if self.do_tanh_gen:
            model.add(Activation("tanh"))
        elif self.fix_delphes_outputs:
            model.add(Lambda(fix_outputs,
                input_shape=self.output_shape, 
                output_shape=self.output_shape
                ))

        model.summary()

        noise = Input(shape=self.noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        ## Head
        if self.add_invmass_disc:
            model.add(Lambda(add_invmass_from_8cartesian,
                input_shape=self.output_shape, 
                output_shape=(self.output_shape[0]+1,), 
                ))
        else:
            model.add(Dense(128,input_shape=self.output_shape))

        if self.do_batch_normalization_disc:
            model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        if self.do_concatenate_disc:
            model.add(Lambda(lambda x: K.concatenate([x*x,x])))
            model.add(LeakyReLU(alpha=0.2))

        ## Main Body
        if self.depth_disc > 0 and self.width_disc > 0:
            for level in xrange(0,self.depth_disc): 
                model.add(Dense(width_disc/(2**level))) #Triangle with width halved at each level
                model.add(LeakyReLU(alpha=0.2))
        else:
            model.add(Dense(128))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(128))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(128))
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

        ## Tail
        model.add(Dense(1,activation='sigmoid'))
        model.summary()

        img = Input(shape=self.output_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self):

        # data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)


        if self.use_delphes:
            data = np.load(self.input_file)
            X_train = data.view((np.float32, len(data.dtype.names)))[:,range(1,22)]
            self.data_ref = X_train[:self.ntest_samples][:,range(1,1+8)]
        else:
            data = np.load(self.input_file)
            X_train = data[:,range(1,1+8)]
            if self.use_ptetaphi_additionally:
                X_train = np.c_[X_train, cartesian_to_ptetaphi(X_train)]
            self.data_ref = X_train[:self.ntest_samples]

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

        # # NOTE. StandardScaler should be fit on training set
        # # and applied the same to train and test, otherwise we 
        # # introduce a bias
        if self.scaler:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train).astype(np.float32)
            pickle.dump(self.scaler, open("progress/{}/scaler.pkl".format(self.tag),'w'))

        half_batch = int(self.batch_size / 2)

        prev_gen_loss = -1
        prev_disc_loss = -1
        n_loss_same_gen = 0  # number of epochs for which generator loss has remained ~same (within 0.01%)
        n_loss_same_disc = 0  # number of epochs for which discriminator loss has remained ~same (within 0.01%)
        for epoch in range(self.nepochs_max):
            
            if n_loss_same_gen > 500 or n_loss_same_disc > 500:
                print "BREAKING because disc/gen loss has remained the same for {}/{} epochs!".format(n_loss_same_disc,n_loss_same_gen)
                break

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            if self.noise_type == 1: # nominal
                noise_half = np.random.normal(0, 1, (half_batch, self.noise_shape[0]))
                noise_full = np.random.normal(0, 1, (self.batch_size, self.noise_shape[0]))

                if epoch % self.nepochs_dump_pred_metrics == 0 and epoch > 0:
                    noise_test = np.random.normal(0, 1, (self.ntest_samples,self.noise_shape[0]))

            elif self.noise_type == 2: # random soup, 4,2,2 have to be modified to sum to noise_shape[0]
                ngaus = self.noise_shape[0] // 2
                nflat = (self.noise_shape[0] - ngaus) // 2
                nexpo = self.noise_shape[0] - nflat - ngaus
                noise_gaus = np.random.normal( 0, 1, (half_batch+self.batch_size, ngaus))
                noise_flat = np.random.uniform(-1, 1, (half_batch+self.batch_size, nflat))
                noise_expo = np.random.exponential( 1,    (half_batch+self.batch_size, nexpo))
                noise = np.c_[ noise_gaus,noise_flat,noise_expo ]
                noise_half = noise[:half_batch]
                noise_full = noise[-self.batch_size:]

                if epoch % self.nepochs_dump_pred_metrics == 0 and epoch > 0:
                    noise_gaus = np.random.normal( 0, 1, (self.ntest_samples, ngaus))
                    noise_flat = np.random.uniform(-1, 1, (self.ntest_samples, nflat))
                    noise_expo = np.random.exponential( 1,    (self.ntest_samples, nexpo))
                    noise_test = np.c_[ noise_gaus,noise_flat,noise_expo ]

            elif self.noise_type == 3: # truth conditioned?
                noise_half = np.c_[ 
                        X_train[np.random.randint(0, X_train.shape[0], half_batch)], 
                        np.random.normal(0, 1, (half_batch,self.noise_shape[0]-X_train.shape[1]))
                        ]
                noise_full = np.c_[ 
                        X_train[np.random.randint(0, X_train.shape[0], self.batch_size)], 
                        np.random.normal(0, 1, (self.batch_size,self.noise_shape[0]-X_train.shape[1]))
                        ]

                if epoch % self.nepochs_dump_pred_metrics == 0 and epoch > 0:
                    noise_test = np.c_[ 
                            X_train[np.random.randint(0, X_train.shape[0], self.ntest_samples)], 
                            np.random.normal(0, 1, (self.ntest_samples,self.noise_shape[0]-X_train.shape[1]))
                            ]

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise_half)

            # Train the discriminator
            ones = np.ones((half_batch, 1))
            zeros = np.zeros((half_batch, 1))

            if self.do_soft_labels:
                ones *= 0.9

            if self.do_noisy_labels:
                frac = 0.3*np.exp(-epoch/self.nepochs_decay_noisy_labels)
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
            valid_y = np.array([1] * self.batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise_full, valid_y)

            if (g_loss - prev_gen_loss) < 0.0001: n_loss_same_gen += 1
            else: n_loss_same_gen = 0
            prev_gen_loss = g_loss

            if (d_loss[0] - prev_disc_loss) < 0.0001: n_loss_same_disc += 1
            else: n_loss_same_disc = 0
            prev_disc_loss = d_loss[0]

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % self.nepochs_dump_pred_metrics == 0 and epoch > 0:

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

                print list(gen_imgs[0])
                # np.save("progress/{}/pred_{}.npy".format(self.tag,epoch), gen_imgs)
                pickle.dump(self.d_epochinfo, open("progress/{}/history.pkl".format(self.tag),'w'))

            if epoch % self.nepochs_dump_history == 0 and epoch > 0:
                fname = "progress/{}/history.pkl".format(self.tag)
                # watch.update(fname)

            if epoch % self.nepochs_dump_models == 0 and epoch > 0: 
                self.discriminator.save("progress/{}/disc_{}.weights".format(self.tag,epoch))
                self.generator.save("progress/{}/gen_{}.weights".format(self.tag,epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    
    parser.add_argument("tag", help="run tag for bookkeping in progress dir")
    # parser.add_argument("--tag", help="run tag for bookkeping in progress dir", default="vtest")
    parser.add_argument("--input_file", help="input numpy file", default="data_xyz.npy")
    parser.add_argument("--output_size", help="size of an element to generate", default=8)
    parser.add_argument("--noise_size", help="size of noise for generator", default=8)
    parser.add_argument("--noise_type", help="noise type (1: gaussians, 2: mixture, 3: truth-conditioned)", default=1)
    parser.add_argument("--ntest_samples", help="number of test samples to dump out", default=5000)
    parser.add_argument("--nepochs_dump_pred_metrics", help="after how many epochs to dump test samples and metrics", default=250)
    parser.add_argument("--nepochs_dump_history", help="after how many epochs to dump history pickle and update web interface", default=500)
    parser.add_argument("--nepochs_dump_models", help="after how many epochs to dump disc and gen models", default=5000)
    parser.add_argument("--nepochs_max", help="max number of epochs to run", default=30001)
    parser.add_argument("--batch_size", help="batch size", default=200)
    parser.add_argument("--do_concatenate_disc", help="concatenation layer of x^2 for discriminator", action="store_true")
    parser.add_argument("--do_concatenate_gen", help="concatenation layer of x^2 for generator", action="store_true")
    parser.add_argument("--do_batch_normalization_disc", help="batch normalization for discriminator", action="store_true")
    parser.add_argument("--do_batch_normalization_gen", help="batch normalization for generator", action="store_true")
    parser.add_argument("--do_soft_labels", help="use 0.9 instead of 1.0 for target values in discriminator training", action="store_true")
    parser.add_argument("--do_noisy_labels", help="flip target values in discriminator training randomly", action="store_true")
    parser.add_argument("--do_tanh_gen", help="use tanh squashing for output of generator", action="store_true")
    parser.add_argument("--nepochs_decay_noisy_labels", help="characteristic decay time in nepochs for noisy label flipping", default=3000)
    parser.add_argument("--use_ptetaphi_additionally", help="instead of 8 cartesian inputs, use 8 cartesian and 8 e pt eta phi (modify output_size to be 16 then)", action="store_true")
    parser.add_argument("--scaler_type", help="type of scaling ('minmax', 'robust', 'standard'). default is none.", default="")
    parser.add_argument("--optimizer_disc", help="optimizer for discriminator (adadelta, adam, sgd, etc)", default="adadelta")
    parser.add_argument("--optimizer_gen", help="optimizer for generator (adadelta, adam, sgd, etc)", default="adadelta")
    parser.add_argument("--beefy_generator", help="Beefy generator architecture", action="store_true")
    parser.add_argument("--depth_gen", help="Number of layers in generator body, for auto generating network architectures.", default=0)
    parser.add_argument("--width_gen", help="Max layer width in generator body, for auto generating network architectures.", default=0)
    parser.add_argument("--depth_disc", help="Number of layers in discriminator body, for auto generating network architectures.", default=0)
    parser.add_argument("--width_disc", help="Max layer width in discriminator body, for auto generating network architectures.", default=0)
    parser.add_argument("--add_invmass_disc", help="Use lambda layer to calculate and include invariant mass in discriminator", action="store_true")
    parser.add_argument("--fix_delphes_outputs", help="Use lambda layer at end to round generator outputs to discrete values for charge, nvtxs, etc", action="store_true")
    parser.add_argument("--use_delphes", help="Use delphes input", action="store_true")
    args = parser.parse_args()

    if args.use_delphes:
        print "Using delphes, so hardcoding output_size to 21!!"
        args.output_size = 21

    if args.use_ptetaphi_additionally:
        args.output_size = 2*int(args.output_size)

    gan = GAN(args)
    gan.train()
