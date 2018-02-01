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
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras import regularizers
from sklearn.preprocessing import *
import numpy as np
np.random.seed(42)
np.set_printoptions(threshold=10,linewidth=150)

import argparse
import time
import pickle
import sys
import os
sys.path.append("../")

from physicsfuncs import Minv

def train_test_split(*args,**kwargs):
    train_size = 1.-kwargs.get("test_size", 0.5)
    for arg in args:
        n_total = arg.shape[0]
        n_train = int(train_size*n_total)
        train = arg[:n_train]
        test = arg[n_train-n_total:]
        yield train
        yield test

class AE():
    def __init__(self, args):

        self.args = args

        self.tag = args.tag

        self.input_file = str(args.input_file)
        self.output_shape = (int(args.output_size),)
        self.nepochs_max = int(args.nepochs_max)
        self.batch_size = int(args.batch_size)
        self.neck_dim = int(args.neck_dim)
        self.node_geometry = str(args.node_geometry)
        self.activation = str(args.activation)
        self.output_activation = str(args.output_activation)
        self.optimizer = str(args.optimizer)
        self.loss = str(args.loss)


        self.scaler_type = args.scaler_type
        self.scaler = None
        if args.scaler_type.lower() == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1.,1.))
        elif args.scaler_type.lower() == "robust":
            self.scaler = RobustScaler()
        elif args.scaler_type.lower() == "standard":
            self.scaler = StandardScaler()

        os.system("mkdir -p progress/{}/".format(self.tag))
        os.system("cp gan.py progress/{}/".format(self.tag))

        self.data_ref = None
        self.d_epochinfo = {}

        self.ae, self.encoder, self.decoder = self.get_models()

    def get_models(self):
        # define the autoencoder
        output_activation = "sigmoid"
        # define the model
        inputs = Input(shape=self.output_shape)
        # "encoded" is the encoded representation of the input

        # upper_geom = [
        #         {"nodes": 50, "activation": "relu"},
        #         {"nodes": 20, "activation": "relu"},
        #         ]
        upper_geom = []
        activation = self.activation
        for nodes in self.node_geometry.split(","):
            upper_geom.append({
                "nodes": int(nodes),
                "activation": activation,
                })
        print upper_geom

        encoded = None
        decoded = None
        for igeom,geom in enumerate(upper_geom):
            to_wrap = inputs if igeom == 0 else encoded
            encoded = Dense(geom["nodes"], activation=geom["activation"])(to_wrap)
        encoded = Dense(self.neck_dim , activation=activation, name="neck")(encoded)
        for igeom,geom in enumerate(upper_geom[::-1]):
            to_wrap = encoded if igeom == 0 else decoded
            decoded = Dense(geom["nodes"], activation=geom["activation"])(to_wrap)
        decoded = Dense(self.output_shape[0] , activation=output_activation)(decoded)

        # AUTOENCODER
        autoencoder = Model(inputs, decoded)
        autoencoder.summary()


        # ENCODER
        encoder = Model(inputs, encoded)

        # DECODER
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.neck_dim,))

        decoder_layer = None
        for ilay in range(-len(upper_geom)-1,0):
            to_wrap = encoded_input if ilay == -len(upper_geom)-1 else decoder_layer
            decoder_layer = autoencoder.layers[ilay](to_wrap)

        decoder = Model(encoded_input, decoder_layer)

        return autoencoder, encoder, decoder

    def get_deltamll(self,samps):
        decoded = self.decoder.predict(self.encoder.predict(samps))
        if self.scaler:
            samps_unscaled = self.scaler.inverse_transform(samps)
            decoded_unscaled = self.scaler.inverse_transform(decoded)
        else:
            samps_unscaled = samps
            decoded_unscaled = decoded
        return Minv(decoded_unscaled)-Minv(samps_unscaled)

    def print_comparison(self,samps):
        encoded = self.encoder.predict(samps)
        decoded = self.decoder.predict(self.encoder.predict(samps))
        if self.scaler:
            samps_unscaled = self.scaler.inverse_transform(samps)
            decoded_unscaled = self.scaler.inverse_transform(decoded)
        else:
            samps_unscaled = samps
            decoded_unscaled = decoded
        print "scaled> sample:",samps
        print "scaled> encoded:",encoded
        print "scaled> decoded encoded:",decoded
        print "scaled> decoded encoded - original:",decoded-samps
        print "unscaled> sample:",samps_unscaled
        print "unscaled> decoded encoded:",decoded_unscaled
        print "unscaled> decoded encoded - original:",decoded_unscaled-samps_unscaled
        print "unscaled> original minv:",Minv(samps_unscaled)
        print "unscaled> decoded minv:",Minv(decoded_unscaled)

    def train(self):

        data = np.load(self.input_file)
        X_total = data[:,range(1,1+8)]
        print X_total

        if self.scaler:
            self.scaler.fit(X_total)
            X_total = self.scaler.transform(X_total).astype(np.float32)
            pickle.dump(self.scaler, open("progress/{}/scaler.pkl".format(self.tag),'w'))

        X_train, X_val = train_test_split(X_total, test_size=0.3)

        # this is our input placeholder
        input_img = Input(shape=self.output_shape)

        self.ae.compile(optimizer=self.optimizer, loss=self.loss)
        print self.ae.summary()

        self.print_comparison(X_val[:3])

        history = self.ae.fit(X_train, X_train,
                epochs = self.nepochs_max,
                batch_size = self.batch_size,
                shuffle = True,
                validation_data = (X_val, X_val))

        self.print_comparison(X_val[:3])

        deltamlls = self.get_deltamll(X_val[:30])

        info = {}
        info["args"] = vars(self.args)
        info["history"] = history.history
        info["deltamlls"] = deltamlls
        info["mu_deltamlls"] = deltamlls.mean()
        info["sig_deltamlls"] = deltamlls.std()
        print info


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("tag", help="run tag for bookkeping in progress dir")
    parser.add_argument("--scaler_type", help="type of scaling ('minmax', 'robust', 'standard'). default is none.", default="")
    parser.add_argument("--nepochs_max", help="max number of epochs to run", default=2)
    parser.add_argument("--input_file", help="input numpy file", default="data_xyz.npy")
    parser.add_argument("--batch_size", help="batch size", default=100)
    parser.add_argument("--output_size", help="size of an element to generate", default=8)
    parser.add_argument("--neck_dim", help="size of encoding layer", default=10)
    parser.add_argument("--node_geometry", help="string of encoding layer scheme (e.g., \"50,20\" to have Dense(50) and Dense(20) before the encoding layer", default="50,20")
    parser.add_argument("--activation", help="activation for all but output", default="relu")
    parser.add_argument("--output_activation", help="activation for output", default="sigmoid")
    parser.add_argument("--optimizer", help="optimizer", default="adadelta")
    parser.add_argument("--loss", help="loss", default="mse")
    
    args = parser.parse_args()

    AE(args).train()
