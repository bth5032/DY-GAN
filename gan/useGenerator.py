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

truthmasses = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)[:,0]
print(truthmasses)
h2 = r.TH1F("h2","masses",60,60,120)
plu.fast_fill(h2, truthmasses)
#for val in truthmasses: h2.Fill(val)
h2.Scale(1./h2.Integral())

for i in xrange(10000,100001, 10000):
  model = load_model("gen_%d.weights" % i)
  model.summary()
  noise = np.random.normal(0, 1, (50000,8))
  #print(noise)
  gen_imgs = model.predict(noise)
  img = gen_imgs
  masses=M(img[:,0],img[:,1],img[:,2],img[:,3],img[:,4],img[:,5],img[:,6],img[:,7])

  print(masses)

  print("mean: %s, std: %s " % (masses.mean(), masses.std()))

  h1 = r.TH1F("h1","masses",60,60,120)
  plu.fast_fill(h1, masses)
  #for val in masses: h1.Fill(val)
  
  h1.Scale(1./h1.Integral())

  ply.plot_hist(
      bgs=[h1,h2],
      legend_labels = ["pred", "truth"],
      options = {
  	"do_stack": False,
  	"yaxis_log": False,
  	#"ratio_numden_indices": [1,0],
  	"output_name": "masses_epoch_%d.pdf" % i,
  	}
      )