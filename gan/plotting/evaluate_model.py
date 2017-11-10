import sys
sys.path.extend(["../../","../"])

import os
import numpy as np
from physicsfuncs import *
import glob
import pickle

import ROOT as r
from keras.models import Sequential, Model, load_model


import plottery.plottery as ply
import plottery.utils as plu

def get_quantities(data):
    masses = Minv(data)
    return {
            "mll": masses[np.isfinite(masses)],
            "ZpZ": Z_pZ(data),
            "ZpT": Z_pT(data),
            "phi": get_phis(data),
            "dphi": get_dphis(data),
            "px": get_pxs(data),
            "py": get_pys(data),
            "pt": get_pts(data),
            "eta": get_etas(data),
            "deta": get_detas(data),
            }

Ngen = 100000

# model_fname = "../gen_90000.weights"
model_fname = "../progress/vnonoise/gen_10000.weights"
model = load_model(model_fname)
data = np.load("../data_xyz.npy")
real = data[:,range(1,1+8)]
print model.summary()

old_noise = True
if old_noise:
    input_shape = (8,)
    noise = np.random.normal(0, 1, (Ngen,input_shape[0]))
else:
    input_shape = (16,)
    noise_gaus = np.random.normal( 0, 1, (Ngen, 8))
    noise_flat = np.random.uniform(-1, 1, (Ngen, 4))
    noise_expo = np.random.exponential( 1,    (Ngen, 4))
    noise = np.c_[ noise_gaus,noise_flat,noise_expo ]


gen = model.predict(noise)

# ss = pickle.load(open('../scaler.sav', 'rb'))
# gen = ss.inverse_transform(gen)

quantities_gen = get_quantities(gen)
quantities_real = get_quantities(real)

label_map = {
            "mll": "m_{ll}",
            "ZpZ": "p_{z}(Z)",
            "ZpT": "p_{T}(Z)",
            "phi": "#phi(lep)",
            "dphi": "#Delta#phi(lep1,lep2)",
            "px": "p_{x}(lep)",
            "py": "p_{y}(lep)",
            "pt": "p_{T}(lep)",
            "eta": "#eta(lep)",
            "deta": "#Delta#eta(lep1,lep2)",
        }

for key in quantities_gen.keys():

    pred = quantities_gen[key]
    truth = quantities_real[key]
    hpred = r.TH1F("hpred","hpred",100,truth.mean()-3.*truth.std(),truth.mean()+3.*truth.std())
    htruth = r.TH1F("htruth","htruth",100,truth.mean()-3.*truth.std(),truth.mean()+3.*truth.std())
    plu.fill_fast(hpred, pred)
    plu.fill_fast(htruth, truth)

    hpred.Scale(1./hpred.Integral())
    htruth.Scale(1./htruth.Integral())

    ply.plot_hist(
            bgs = [htruth,hpred],
            colors = [r.kAzure+2,r.kRed-2],
            legend_labels = ["Real", "Fake"],
            options = {
                "bkg_sort_method": "unsorted",
                "do_stack": False,
                "legend_percentageinbox": False,
                "legend_alignment": "bottom right",
                "legend_scalex": 0.7,
                "legend_scaley": 0.7,
                "legend_border": False,
                "xaxis_label": label_map[key],
                "yaxis_label": "Entries",
                "output_name": "plots/compare_{}.pdf".format(key),
                "ratio_numden_indices": [1,0],
                "output_ic": True,
                }
            )

