import numpy as np
import keras
import keras.backend.tensorflow_backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.layers.advanced_activations import LeakyReLU

def Mass(x):
    return [K.sqrt((x[:,0]+x[:,4])**2 - (x[:,1]+x[:,5])**2 - (x[:,2]+x[:,6])**2 - (x[:,3]+x[:,7])**2)]

x = Input(shape=(8,))
out = Lambda(
        lambda x: K.concatenate([x,Mass(x)])
        )(x)
# invariant mass of below cartesian stuff is 91.73560874
inputs = np.array([[
    52.72488   ,
    21.419755  ,
    40.11684   ,
    26.678367  ,
    57.297302  ,
    -30.0209    ,
    -35.68861   ,
    33.287204  ]])
print inputs

model = Model([x], [out])
model.compile('adam', 'mse')
print model.predict(inputs)


a=Sequential()
a.add(Lambda(
        lambda x: K.concatenate([x,Mass(x)]),
        input_shape = x, output_shape=(9,)
        ))
a.add(Dense(128))
a.add(LeakyReLU(alpha=0.2))

model2 = Model([x], a([x]))
model2.compile('adam', 'mse')
model2.train_on_batch()
print model2.predict(inputs)

