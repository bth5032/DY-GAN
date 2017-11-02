import numpy as np
from physicsfuncs import M
import glob

import ROOT as r

import sys
sys.path.append("../")

import plottery.plottery as ply
import plottery.utils as plu

def get_quantities(fname):
    data = np.load(fname)
    masses = M(data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7])
    return {
            "epoch": int(fname.rsplit("_",1)[-1].split(".")[0]),
            "masses": masses[np.isfinite(masses)],
            }

# fname = "progress/pred_80.npy"
fnames = glob.glob("progress/*npy")
points = []
for fname in fnames:
    quantities = get_quantities(fname)
    if not np.isfinite(quantities["masses"].mean()): continue
    if not np.isfinite(quantities["masses"].std()): continue
    points.append([quantities["epoch"], quantities["masses"].mean(), quantities["masses"].std()*2.35])

points = sorted(points)

points = np.array(points)
xvals = points[:,0]
yvals = points[:,1]
ydown = points[:,2]
yup = points[:,2]

ply.plot_graph(
        [
            ([0.,3500.],[91.2,91.2],[2.5/2,2.5/2],[2.5/2,2.5/2]),
            (xvals,yvals,ydown,yup),
            ],
        colors = [r.kAzure+2,r.kRed-2], #, r.kGreen-2, r.kAzure+2],
        legend_labels = ["m_{Z}","#mu(inv. mass)"], #, "green", "blue"],
        options = {
            "legend_alignment": "bottom right",
            "legend_scalex": 0.7,
            "xaxis_label": "epoch",
            "yaxis_label": "#mu(inv. mass)",
            "output_name": "test.pdf",
            "output_ic": True,
            }
        )
