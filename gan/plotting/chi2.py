import sys
sys.path.extend(["../../","../"])

import os
import numpy as np
from physicsfuncs import *
import glob

import ROOT as r


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
            "pt": get_pts(data),
            "eta": get_etas(data),
            "deta": get_detas(data),
            }

# fnames = glob.glob("../progress/adam40k/*npy")
fnames = glob.glob("../progress/vtest/*npy")
data = np.load("../data_xyz.npy")

quantities_real = get_quantities(data[:,range(1,9)])
epoch_chi2 = []

fnames = sorted(fnames, key=lambda x:int(x.rsplit("_",1)[-1].split(".")[0]))
for fname in fnames:
    data = np.load(fname)
    epoch = int(fname.rsplit("_",1)[-1].split(".")[0])
    if epoch < 500: continue
    quantities_gen = get_quantities(data)
    tot_chi2 = 0.
    for key in quantities_gen:
        tot_chi2 += get_redchi2(quantities_real[key], quantities_gen[key])
    print epoch, tot_chi2
    epoch_chi2.append([epoch, tot_chi2])
epoch_chi2 = np.array(epoch_chi2)

ply.plot_graph(
    [
        (epoch_chi2[:,0],epoch_chi2[:,1])
        ],
    colors = [r.kAzure+2],
    legend_labels = ["model1"],
    options = {
        "legend_alignment": "top right",
        "legend_scalex": 0.7,
        "xaxis_label": "epoch",
        "yaxis_label": "#Sigma(#chi^{2}/ndof)",
        "yaxis_log": True,
        "title": "#Sigma(#chi^{2}/ndof) vs epoch",
        "output_name": "plots/chi2_vs_epoch.pdf",
        "output_ic": True,
        }
    )

