import sys
sys.path.extend(["../../","../"])
import os
import numpy as np
import glob
import pickle
import ROOT as r
import plottery.plottery as ply

def ema(a, alpha=0.30, windowSize=5):
    wghts = (1-alpha)**np.arange(windowSize)
    wghts /= wghts.sum()
    out = np.convolve(a,wghts)
    out[:windowSize-1] = a[:windowSize-1]
    return out[:a.size] 

class Model(object):
    def __init__(self, d, name):
        self.d = d
        self.name = name

    def get_args(self):
        return self.d["args"]

    def get_points(self, key, smooth=True):
        if key == "time":
            points = np.array(zip(self.d["epoch"], self.d[key]))
            points[:,1] -= points[0][1]
            points[:,1] /= 60.
            if smooth: points[:,1] = ema(points[:,1])
            return points.T
        else:
            points = np.array(zip(self.d["epoch"], self.d[key]))
            points[np.isnan(points)] = 0.
            if smooth: points[:,1] = ema(points[:,1])
            return points.T

modelinfos = [
        ["../progress/vdelphes/","vdelphes"],
        ["../progress/vdelphes/","vdelphes"],
        ]

models = []
for modeldir,name in modelinfos:
    fname_history = "{}/history.pkl".format(modeldir)
    if not os.path.exists(fname_history): continue
    data = pickle.load(open(fname_history,"r"))
    models.append( Model(data,name) )

varinfo = [
        ["d_loss", "discriminator loss"],
        ["g_loss", "generator loss"],
        ["d_acc", "discriminator accuracy"],
        ["mass_mu", "m_{ll} mean"],
        ["mass_sig", "m_{ll} width"],
        ["time", "cumulative time [minutes]"],
        ]
for varname, varlabel in varinfo:
    ply.plot_graph(
        [m.get_points(varname) for m in models],
        legend_labels = [m.name for m in models],
        options = {
            "legend_alignment": "top left",
            # "legend_scalex": 0.7,
            "xaxis_label": "epoch",
            "yaxis_label": varlabel,
            "output_name": "plots/{}.pdf".format(varname),
            "output_ic": True,
            }
        )

