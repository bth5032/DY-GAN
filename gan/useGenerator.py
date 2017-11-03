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

model_name=None
ngen_samples=50000
input_size=(23,)
truth_data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)
truth_masses = truth_data[:,0]
truth_4vecs = truth_data[:,range(1,1+8)]

print(truth_masses)
h2 = r.TH1F("h2","masses",60,60,120)
#plu.fill_fast(h2, truthmasses)
for val in truth_masses: h2.Fill(val)
h2.Scale(1./h2.Integral())

#Check for a model in model/ (a directory starting with model_)
if (os.listdir("model/")):
  model_name=filter(lambda x: "model_" in x, os.listdir("model/"))[0]

#If no directory exists, exit, no model can be found
if not model_name:
  print("No model found in model/, please copy one over from old_models or run gan.py to generate a model")
  exit(1)
else:
  print("Making plots for model in directory: %s" % model_name)

#The one liner is a bit ugly, but it just gets a list of full paths to the gen_*.weights files inside of model/model_<model_name>/
#i.e. it finds all the models we want to run over...
#for saved_model in ["model/model_TruthBiased1/gen_10000.weights", "model/model_TruthBiased1/gen_20000.weights"]: #map(lambda y: "model/"+model_name+"/"+y, filter(lambda x: ("gen_" in x and ".weights" in x), os.listdir("model/"+model_name))):
for saved_model in map(lambda y: "model/"+model_name+"/"+y, filter(lambda x: ("gen_" in x and ".weights" in x), os.listdir("model/"+model_name))):
  model = load_model(saved_model)
  model.summary()
  
  noise = np.random.normal(0, 1, (ngen_samples,input_size[0]-truth_4vecs.shape[1]))
  rand_rows = np.random.randint(0, truth_4vecs.shape[0], ngen_samples)
  gen_input = np.c_[noise, truth_4vecs[rand_rows]]
  print(gen_input.shape)
  gen_output = model.predict(gen_input)
  print(gen_output.shape)
  masses=M(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])

  print(masses)

  print("mean: %s, std: %s " % (masses.mean(), masses.std()))

  h1 = r.TH1F("h1","masses",60,60,120)
  #plu.fill_fast(h1, masses)
  for val in masses: h1.Fill(val)
  
  h1.Scale(1./h1.Integral())

  ply.plot_hist(
      bgs=[h1,h2],
      legend_labels = ["pred", "truth"],
      options = {
  	"do_stack": False,
  	"yaxis_log": False,
  	#"ratio_numden_indices": [1,0],
  	"output_name": "%s/masses_epoch_%s.pdf" % (saved_model[:saved_model.index("gen_")], saved_model[saved_model.index("gen_")+4:saved_model.index(".weights")]),
  	}
      )