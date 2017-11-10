import os
import numpy as np
from physicsfuncs import *
import glob

import ROOT as r

import sys
sys.path.append("../")

import plottery.plottery as ply
import plottery.utils as plu

def get_quantities(fname):
    data = np.load(fname)
    masses = Minv(data)
    return {
            "epoch": int(fname.rsplit("_",1)[-1].split(".")[0]),
            "masses": masses[np.isfinite(masses)],
            "Z_pZ": Z_pZ(data),
            "Z_pT": Z_pT(data),
            "phis": get_phis(data),
            "dphis": get_dphis(data),
            "etas": get_etas(data),
            "detas": get_detas(data),
            # eta
            # phi
            # delta eta
            # delta phi
            }

# fname = "progress/pred_80.npy"
# fnames = glob.glob("progress/vspherical/*npy")
fnames = glob.glob("progress/vtestadam/*npy")
# fnames = glob.glob("progress/*npy")
points_mz = []
points_zpt = []
points_zpz = []
points_phis = []
points_dphis = []
points_etas = []
points_detas = []
for fname in fnames:
    quantities = get_quantities(fname)
    if not np.isfinite(quantities["masses"].mean()): continue
    if not np.isfinite(quantities["masses"].std()): continue

    points_mz.append([quantities["epoch"], quantities["masses"].mean(), quantities["masses"].std()])
    points_zpt.append([quantities["epoch"], quantities["Z_pT"].mean(), quantities["Z_pT"].std()])
    points_zpz.append([quantities["epoch"], quantities["Z_pZ"].mean(), quantities["Z_pZ"].std()])
    points_phis.append([quantities["epoch"], quantities["phis"].mean(), quantities["phis"].std()])
    points_dphis.append([quantities["epoch"], quantities["dphis"].mean(), quantities["dphis"].std()])
    points_etas.append([quantities["epoch"], quantities["etas"].mean(), quantities["etas"].std()])
    points_detas.append([quantities["epoch"], quantities["detas"].mean(), quantities["detas"].std()])

data = np.load("data_xyz.npy")
mZs = data[:,0]

zpz = Z_pZ(data[:,range(1,9)])
zpt = Z_pT(data[:,range(1,9)])
phis = get_phis(data[:,range(1,9)])
dphis = get_dphis(data[:,range(1,9)])
etas = get_etas(data[:,range(1,9)])
detas = get_detas(data[:,range(1,9)])

# zcent = 90.9925
# zstd = 5.2383


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def make_plot(points, truth, label_truth, label_pred, fname):
    truth_cent = truth.mean()
    truth_std = truth.std()
    points = sorted(points)

    smooth = True
    window = 15
    points = np.array(points)
    if not smooth:
        xvals = points[:,0]
        yvals = points[:,1]
        ydown = points[:,2]
        yup = points[:,2]
    else:
        xvals = points[:,0][window-1:]
        yvals = moving_average(points[:,1],n=window)
        ydown = moving_average(points[:,2],n=window)
        yup = moving_average(points[:,2],n=window)

    # hpred = r.TH1F("hpred",100,truth.min(),truth.max())
    # htruth = r.TH1F("htruth",100,truth.min(),truth.max())
    # fill_fast(hpred, yvals)
    # fill_fast(htruth, truth)

    ply.plot_graph(
            [
                ([0.,max(xvals)],[truth_cent,truth_cent],[truth_std,truth_std],[truth_std,truth_std]),
                (xvals,yvals,ydown,yup),
                ],
            colors = [r.kAzure+2,r.kRed-2],
            legend_labels = [label_truth, label_pred],
            options = {
                "legend_alignment": "bottom right",
                "legend_scalex": 0.7,
                "xaxis_label": "epoch",
                "yaxis_label": label_pred,
                "output_name": fname,
                "output_ic": True,
                }
            )

make_plot(points_mz, mZs, "m_{Z}", "#mu(inv. mass)", "test.png")
make_plot(points_zpt, zpt, "p_{T}^{Z}", "p_{T}^{Z} generated", "test.png")
make_plot(points_zpz, zpz, "p_{z}^{Z}", "p_{z}^{Z} generated", "test.png")
make_plot(points_phis, phis, "#phi(lep)", "#phi(lep) generated", "test.png")
make_plot(points_dphis, dphis, "#delta#phi(l1,l2)", "#delta#phi(l1,l2) generated", "test.png")
make_plot(points_etas, etas, "#eta(lep)", "#eta(lep) generated", "test.png")
make_plot(points_detas, detas, "#delta#eta(l1,l2)", "#delta#eta(l1,l2) generated", "test.png")
