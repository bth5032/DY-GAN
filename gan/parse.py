import numpy as np
from physicsfuncs import *
import glob

import ROOT as r

import sys, os
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
            # eta
            # phi
            # delta eta
            # delta phi
            }

# fname = "progress/pred_80.npy"
# fnames = glob.glob("progress/v1/*npy")

model_name=None
#Check for a model in model/ (a directory starting with model_)
if (os.listdir("model/")):
  model_name=filter(lambda x: "model_" in x, os.listdir("model/"))[0]
#If no directory exists, exit, no model can be found
if not model_name:
  print("No model found in model/, please copy one over from old_models or run gan.py to generate a model")
  exit(1)
else:
  print("Making plots for model in directory: %s" % model_name)
fnames=map(lambda y: "model/"+model_name+"/"+y, filter(lambda x: (".npy" in x), os.listdir("model/"+model_name)))


# fnames = glob.glob("progress/*npy")
points_mz = []
points_zpt = []
points_zpz = []
points_phis = []
points_dphis = []
for fname in fnames:
    quantities = get_quantities(fname)
    if not np.isfinite(quantities["masses"].mean()): continue
    if not np.isfinite(quantities["masses"].std()): continue

    points_mz.append([quantities["epoch"], quantities["masses"].mean(), quantities["masses"].std()])
    points_zpt.append([quantities["epoch"], quantities["Z_pT"].mean(), quantities["Z_pT"].std()])
    points_zpz.append([quantities["epoch"], quantities["Z_pZ"].mean(), quantities["Z_pZ"].std()])
    points_phis.append([quantities["epoch"], quantities["phis"].mean(), quantities["phis"].std()])
    points_dphis.append([quantities["epoch"], quantities["dphis"].mean(), quantities["dphis"].std()])

#data = np.load("data_xyz.npy")
data = np.loadtxt(open("dy_mm_events_line.input", "r"), delimiter=",", skiprows=1)
mZs = data[:,0]

zpz = Z_pZ(data[:,range(1,9)])
zpt = Z_pT(data[:,range(1,9)])
phis = get_phis(data[:,range(1,9)])
dphis = get_dphis(data[:,range(1,9)])

# zcent = 90.9925
# zstd = 5.2383


def make_plot(points, truth_cent, truth_std, label_truth, label_pred, fname):
    points = sorted(points)

    points = np.array(points)
    xvals = points[:,0]
    yvals = points[:,1]
    ydown = points[:,2]
    yup = points[:,2]

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
                #"output_ic": True,
                }
            )

make_plot(points_mz, mZs.mean(),mZs.std(), "m_{Z}", "#mu(inv. mass)", "model/"+model_name+"/dilmass.pdf")
make_plot(points_zpt, zpt.mean(),zpt.std(), "p_{T}^{Z}", "p_{T}^{Z} generated", "model/"+model_name+"/dilpt.pdf")
make_plot(points_zpz, zpz.mean(),zpz.std(), "p_{z}^{Z}", "p_{z}^{Z} generated", "model/"+model_name+"/dilpz.pdf")
make_plot(points_phis, phis.mean(),phis.std(), "#phi(lep)", "#phi(lep) generated", "model/"+model_name+"/lep_phi.pdf")
make_plot(points_dphis, dphis.mean(),dphis.std(), "#delta#phi(l1,l2)", "#delta#phi(l1,l2) generated", "model/"+model_name+"/lep_dphi.pdf")
