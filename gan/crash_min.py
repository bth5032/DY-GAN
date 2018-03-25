import keras.backend.tensorflow_backend as K
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU, Lambda
from keras.layers import Input, Add
from keras.losses import binary_crossentropy
import numpy as np

def get_first_N(x,N=10):
    return x[:,0:N]

class CrashingKeras():

    def __init__(self, **kwargs):
        self.input_shape=(int(10),)
        self.lambda_output_shape=(int(5),)
        self.batch_size=100
        self.nepochs_max=5
        self.NN = self.build_net()
        self.NN.compile(loss="binary_crossentropy", optimizer="adam")
    
    def build_net(self):
        inputs = Input(shape=self.input_shape)
        x = Dense(10)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(5)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #z = Lambda(lambda y: y[:,0:self.lambda_output_shape[0]], output_shape=self.lambda_output_shape)(inputs)
        z=Lambda(get_first_N, arguments={'N': self.lambda_output_shape[0]})(inputs)
        x = Add()([x,z])
        x = Dense(1)(x)
        model = Model(inputs=inputs, outputs=[x])
        return model

    def train(self):        
        noise = np.random.normal(0, 1, (self.batch_size, self.input_shape[0]))
        for epoch in range(self.nepochs_max):
            valid_y = np.array([1] * self.batch_size)
            loss = self.NN.train_on_batch(noise, valid_y)
            self.NN.save("gen_{}.weights".format(epoch))

network = CrashingKeras()

network.train()
