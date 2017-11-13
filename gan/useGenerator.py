from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

import sys, os, argparse
sys.path.append("../")

import plottery.plottery as ply
import plottery.utils as plu
import matplotlib.pyplot as plt

import ROOT as r
import physicsfuncs as pf

import numpy as np

input_size=(23,)

def getTrueDists(truth_4vecs, truth_masses):
  mass   = None
  zpt    = None
  zpx    = None
  zpy    = None
  zpz    = None
  ztheta = None
  zphi   = None

  if ("true_distributions.root" in os.listdir(".")):
    f = r.TFile("true_distributions.root", "READ")
    
    mass   = f.Get("h_true_mass")
    zpt    = f.Get("h_true_zpt")
    zpx    = f.Get("h_true_zpx")
    zpy    = f.Get("h_true_zpy")
    zpz    = f.Get("h_true_zpz")
    ztheta = f.Get("h_true_ztheta")
    zphi   = f.Get("h_true_zphi")

    mass   .SetDirectory(0)
    zpt    .SetDirectory(0)
    zpx    .SetDirectory(0)
    zpy    .SetDirectory(0)
    zpz    .SetDirectory(0)
    ztheta .SetDirectory(0)
    zphi   .SetDirectory(0)

    f.Close()
  else:
    print("No ROOT file for true values found, computing...")

    f = r.TFile("true_distributions.root", "CREATE")
    f.cd()

    mass   = r.TH1F("h_true_mass"   , "masses" ,60,60,120)
    zpt    = r.TH1F("h_true_zpt"    , "zpt"    ,100,0,100)
    zpx    = r.TH1F("h_true_zpx"    , "zpx"    ,200,60,120)
    zpy    = r.TH1F("h_true_zpy"    , "zpy"    ,200,0,200)
    zpz    = r.TH1F("h_true_zpz"    , "zpz"    ,200,0,200)
    ztheta = r.TH1F("h_true_ztheta" , "ztheta" ,63,-3.15,3.15)
    zphi   = r.TH1F("h_true_zphi"   , "zphi"   ,63,-3.15,3.15)

    mass   .SetDirectory(0)
    zpt    .SetDirectory(0)
    zpx    .SetDirectory(0)
    zpy    .SetDirectory(0)
    zpz    .SetDirectory(0)
    ztheta .SetDirectory(0)
    zphi   .SetDirectory(0)

    #plu.fill_fast(h_true_mass, truthmasses)
    for val in truth_masses            : mass.Fill(val)
    for val in pf.Z_pT(truth_4vecs)    : zpt.Fill(val)
    for val in pf.Z_pX(truth_4vecs)    : zpx.Fill(val)
    for val in pf.Z_pY(truth_4vecs)    : zpy.Fill(val)
    for val in pf.Z_pZ(truth_4vecs)    : zpz.Fill(val)
    for val in pf.Z_theta(truth_4vecs) : ztheta.Fill(val)
    for val in pf.Z_phi(truth_4vecs)   : zphi.Fill(val)

    mass   .Scale(1./mass.Integral())
    zpt    .Scale(1./zpt.Integral())
    zpx    .Scale(1./zpx.Integral())
    zpy    .Scale(1./zpy.Integral())
    zpz    .Scale(1./zpz.Integral())
    ztheta .Scale(1./ztheta.Integral())
    zphi   .Scale(1./zphi.Integral())

    mass   .Write()
    zpt    .Write()
    zpx    .Write()
    zpy    .Write()
    zpz    .Write()
    ztheta .Write()
    zphi   .Write()

    print("Saving into true_distributions.root")

    f.Close()
  
  return {"mass": mass, "zpt": zpt, "zpx": zpx, "zpy": zpy, "zpz": zpz, "ztheta": ztheta, "zphi": zphi}

def generatePlots(model_dir, ngen_samples=50000):

  #The one liner is a bit ugly, but it just gets a list of full paths to the gen_*.weights files inside of model/model_<model_name>/
  #i.e. it finds all the models we want to run over...
  #for saved_model in ["model/model_TruthBiased1/gen_10000.weights", "model/model_TruthBiased1/gen_20000.weights"]: #map(lambda y: "model/"+model_name+"/"+y, filter(lambda x: ("gen_" in x and ".weights" in x), os.listdir("model/"+model_name))):
  for saved_model in map(lambda y: model_dir+y, filter(lambda x: ("gen_" in x and ".weights" in x), os.listdir(model_dir))):
    if "gen_0.weights" in saved_model:
      continue

    print("Loading model for %s" % saved_model)
    model = load_model(saved_model)
    model.summary()

    truth_data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)
    truth_masses = truth_data[:,0]
    truth_4vecs = truth_data[:,range(1,1+8)]
    
    noise = np.random.normal(0, 1, (ngen_samples,input_size[0]-truth_4vecs.shape[1]))
    rand_rows = np.random.randint(0, truth_4vecs.shape[0], ngen_samples)
    gen_input = np.c_[noise, truth_4vecs[rand_rows]]
    #print(gen_input.shape)
    gen_output = model.predict(gen_input)
    #print(gen_output.shape)
    
    masses=pf.M(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    #px=pf.Z_pX(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    #py=pf.Z_pY(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    #pz=pf.Z_pZ(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    #theta=pf.Z_theta(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    #phi=pf.Z_phi(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    #pt=pf.Z_pT(gen_output[:,0],gen_output[:,1],gen_output[:,2],gen_output[:,3],gen_output[:,4],gen_output[:,5],gen_output[:,6],gen_output[:,7])
    px=pf.Z_pX(gen_output)
    py=pf.Z_pY(gen_output)
    pz=pf.Z_pZ(gen_output)
    theta=pf.Z_theta(gen_output)
    phi=pf.Z_phi(gen_output)
    pt=pf.Z_pT(gen_output)
    #dphi=pf.get_dphis(gen_output)
    #dtheta=pf.get_dthetas(gen_output)
    #dpt=pf.get_dpT(gen_output)

    print(masses)

    print("mean: %s, std: %s " % (masses.mean(), masses.std()))

    h_mass   = r.TH1F("h_mass_%s"   % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "masses" ,60,60,120)
    h_zpt    = r.TH1F("h_zpt_%s"    % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "zpt"    ,100,0,100)
    h_zpx    = r.TH1F("h_zpx_%s"    % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "zpx"    ,200,60,120)
    h_zpy    = r.TH1F("h_zpy_%s"    % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "zpy"    ,200,0,200)
    h_zpz    = r.TH1F("h_zpz_%s"    % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "zpz"    ,200,0,200)
    h_ztheta = r.TH1F("h_ztheta_%s" % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "ztheta" ,63,-3.15,3.15)
    h_zphi   = r.TH1F("h_zphi_%s"   % saved_model[saved_model.index("gen_"):saved_model.index(".weights")], "zphi"   ,63,-3.15,3.15)
    
    #plu.fill_fast(h1, masses)
    for val in masses: h_mass.Fill(val)
    
    h_mass.Scale(1./h_mass.Integral())
    h_zpt.Scale(1./h_mass.Integral())
    h_zpx.Scale(1./h_mass.Integral())
    h_zpy.Scale(1./h_mass.Integral())
    h_zpz.Scale(1./h_mass.Integral())
    h_ztheta.Scale(1./h_mass.Integral())
    h_zphi.Scale(1./h_mass.Integral())

    true_dists=getTrueDists(truth_4vecs, truth_masses)

    model_name=saved_model[:saved_model.index("gen_")]
    epoch=saved_model[saved_model.index("gen_")+4:saved_model.index(".weights")]

    ply.plot_hist(
        bgs=[h_mass,true_dists["mass"]],
        legend_labels = ["pred", "truth"],
        options = {
    	"do_stack": False,
    	"yaxis_log": False,
    	#"ratio_numden_indices": [1,0],
    	"output_name": "%s/masses_epoch_%s.pdf" % (model_name, epoch),
    	}
        )
    """ply.plot_hist(
        bgs=[h_zpt,true_dists["zpt"]],
        legend_labels = ["pred", "truth"],
        options = {
      "do_stack": False,
      "yaxis_log": False,
      #"ratio_numden_indices": [1,0],
      "output_name": "%s/zpt_epoch_%s.pdf" % (model_name, epoch),
      }
        )
    ply.plot_hist(
        bgs=[h_zpx,true_dists["zpx"]],
        legend_labels = ["pred", "truth"],
        options = {
      "do_stack": False,
      "yaxis_log": False,
      #"ratio_numden_indices": [1,0],
      "output_name": "%s/zpx_epoch_%s.pdf" % (model_name, epoch),
      }
        )
    ply.plot_hist(
        bgs=[h_zpy,true_dists["zpy"]],
        legend_labels = ["pred", "truth"],
        options = {
      "do_stack": False,
      "yaxis_log": False,
      #"ratio_numden_indices": [1,0],
      "output_name": "%s/zpy_epoch_%s.pdf" % (model_name, epoch),
      }
        )
    ply.plot_hist(
        bgs=[h_zpz,true_dists["zpz"]],
        legend_labels = ["pred", "truth"],
        options = {
      "do_stack": False,
      "yaxis_log": False,
      #"ratio_numden_indices": [1,0],
      "output_name": "%s/zpz_epoch_%s.pdf" % (model_name, epoch),
      }
        )
    ply.plot_hist(
        bgs=[h_ztheta,true_dists["ztheta"]],
        legend_labels = ["pred", "truth"],
        options = {
      "do_stack": False,
      "yaxis_log": False,
      #"ratio_numden_indices": [1,0],
      "output_name": "%s/ztheta_epoch_%s.pdf" % (model_name, epoch),
      }
        )
    ply.plot_hist(
        bgs=[h_phi,true_dists["zphi"]],
        legend_labels = ["pred", "truth"],
        options = {
      "do_stack": False,
      "yaxis_log": False,
      #"ratio_numden_indices": [1,0],
      "output_name": "%s/zphi_epoch_%s.pdf" % (model_name, epoch),
      }
        )"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--tag", "-t", help="Tag for the run", type=str)
  parser.add_argument("--dir", "-d", help="Dir to make plots from, overrules tag, used when you want to plot an older model of a tag", type=str)
  parser.add_argument("--ngen", "-n", help="Number of sample points to generate per model", type=int)
  args = parser.parse_args()

  model_name = None
  model_dir = None

  if args.dir:
    model_dir=args.dir
  else:
    if args.tag:
      tag=args.tag
      model_name = sorted(filter(lambda x: ("model_" in x) and (tag in x), os.listdir("model/")))[0]
    else:
      model_name = sorted(filter(lambda x: "model_" in x, os.listdir("model/")))[0]
    model_dir="model/"+model_name+"/"

  if args.ngen:
    ngen=args.ngen
  else:
    ngen=50000


  print("Making plots for model in directory: %s" % model_dir)
  generatePlots(model_dir, ngen)
