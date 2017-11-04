import sys
import numpy as np
from tqdm import tqdm
import ROOT as r
import glob


if len(sys.argv) <= 1:
    raise Exception("[!] Give me a filename or quote-surrounded wildcard pattern")

r.gSystem.Load('CMSSW_9_1_0_pre3/delphes-3.4.2pre03/libDelphes.so')
r.gInterpreter.Declare('#include "CMSSW_9_1_0_pre3/delphes-3.4.2pre03/classes/DelphesClasses.h"')
ch = r.TChain("Delphes")

for fname in glob.glob(sys.argv[-1]):
    print ">>> Adding {}".format(fname)
    ch.Add(fname)

data = []

ch.SetBranchStatus("Event*",0)
ch.SetBranchStatus("Particle*",0)
ch.SetBranchStatus("EFlow*",0)
ch.SetBranchStatus("Photon*",0)
ch.SetBranchStatus("Electron*",0)

ievt = 0
for evt in tqdm(ch,total=ch.GetEntries()):
    ievt += 1
    # if ievt > 1000: break
    if evt.MuonTight_size != 2: continue


    lep1 = evt.MuonTight[0].P4()
    lep2 =  evt.MuonTight[1].P4()
    mll = (lep1+lep2).M()

    if mll < 50. or mll > 150.: continue

    lep1_charge =  evt.MuonTight[0].Charge
    lep1_iso =  evt.MuonTight[0].IsolationVar

    lep2_charge =  evt.MuonTight[1].Charge
    lep2_iso =  evt.MuonTight[1].IsolationVar

    if lep1_charge*lep2_charge > 0: continue # cleans up a small fraction (<0.02%!)

    metphi = evt.MissingET[0].Phi
    met = evt.MissingET[0].MET

    ngenjets = evt.GenJet_size
    nvtxs = evt.Vertex_size

    NMAXJETS = 5
    genjet_pts = []
    for genjet in evt.GenJet:
        genjet_pts.append(genjet.P4().Pt())
    genjet_pts = genjet_pts[:NMAXJETS] + [0.] * (NMAXJETS - len(genjet_pts)) # zero pad

    row = [
            mll, 
            lep1.E(), lep1.Px(), lep1.Py(), lep1.Pz(),
            lep2.E(), lep2.Px(), lep2.Py(), lep2.Pz(),
            lep1_charge, lep1_iso,
            lep2_charge, lep2_iso,
            nvtxs,
            met, metphi, ngenjets,
            ]+genjet_pts

    data.append(row)

colnames = [
        "mll",
        "lep1_e",
        "lep1_px",
        "lep1_py",
        "lep1_pz",
        "lep2_e",
        "lep2_px",
        "lep2_py",
        "lep2_pz",
        "lep1_charge",
        "lep1_iso",
        "lep2_charge",
        "lep2_iso",
        "nvtxs",
        "met",
        "metphi",
        "ngenjets",
        "genjet_pt1",
        "genjet_pt2",
        "genjet_pt3",
        "genjet_pt4",
        "genjet_pt5",
        ]

data = np.array(data,dtype=np.float32)
data = np.core.records.fromarrays(data.transpose(), names=colnames)

data.dump("data.npa")
