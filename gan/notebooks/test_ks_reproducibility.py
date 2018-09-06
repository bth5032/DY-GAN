import hashlib
from scipy.stats import ks_2samp
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
tf.set_random_seed(42)
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
session = tf.Session(config=config)
print "import tensorflow"
           
import keras.backend.tensorflow_backend as K

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Lambda
from keras.layers import Input, merge, Concatenate, concatenate, Add
from keras.losses import binary_crossentropy
from keras.utils import plot_model
print "import keras"

import numpy as np
#from tqdm import tqdm
import time
import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

print "import matplotlib"

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import binned_statistic_2d

print "import sklearn"

np.random.seed(42)

cov_hash = None
cov_ans = None

def Minv(cols,ptetaphi=False,nopy2=False):
    """
    Computes M for two objects given the cartesian momentum projections
    if `ptetaphi` is True, then assumes the 8 input columns are cylindrical eptetaphi
    if `nopy2` is True, input is 7 columns with no py2
    """
    if ptetaphi:
        cols = ptetaphi_to_cartesian(cols)
    if nopy2:
        M2 = (cols[:,0]+cols[:,4])**2
        M2 -= (cols[:,1]+cols[:,5])**2
        M2 -= (cols[:,2]          )**2
        M2 -= (cols[:,3]+cols[:,6])**2
    else:
        M2 = (cols[:,0]+cols[:,4])**2
        M2 -= (cols[:,1]+cols[:,5])**2
        M2 -= (cols[:,2]+cols[:,6])**2
        M2 -= (cols[:,3]+cols[:,7])**2
    return np.sqrt(M2)

def cartesian_to_ptetaphi(eight_cartesian_cols):
    """
    Takes 8 columns as cartesian e px py pz e px py pz
    and converts to e pt eta phi e pt eta phi
    """
    e1 =  eight_cartesian_cols[:,0]
    e2 =  eight_cartesian_cols[:,4]
    px1 = eight_cartesian_cols[:,1]
    px2 = eight_cartesian_cols[:,5]
    py1 = eight_cartesian_cols[:,2]
    py2 = eight_cartesian_cols[:,6]
    pz1 = eight_cartesian_cols[:,3]
    pz2 = eight_cartesian_cols[:,7]
    p1 = np.sqrt(px1**2+py1**2+pz1**2)
    p2 = np.sqrt(px2**2+py2**2+pz2**2)
    pt1 = np.sqrt(px1**2+py1**2)
    pt2 = np.sqrt(px2**2+py2**2)
    phi1 = np.arctan2(py1,px1)
    phi2 = np.arctan2(py2,px2)
    eta1 = np.arctanh(pz1/p1)
    eta2 = np.arctanh(pz2/p2)
    return np.c_[e1,pt1,eta1,phi1,e2,pt2,eta2,phi2]

def ptetaphi_to_cartesian(eight_eptetaphi_cols):
    """
    Takes 8 columns as e pt eta phi e pt eta phi
    and converts to e px py pz e px py pz
    """
    e1 =  eight_eptetaphi_cols[:,0]
    e2 =  eight_eptetaphi_cols[:,4]
    pt1 =  eight_eptetaphi_cols[:,1]
    pt2 =  eight_eptetaphi_cols[:,5]
    eta1 =  eight_eptetaphi_cols[:,2]
    eta2 =  eight_eptetaphi_cols[:,6]
    phi1 =  eight_eptetaphi_cols[:,3]
    phi2 =  eight_eptetaphi_cols[:,7]
    px1 = np.abs(pt1)*np.cos(phi1)
    px2 = np.abs(pt2)*np.cos(phi2)
    py1 = np.abs(pt1)*np.sin(phi1)
    py2 = np.abs(pt2)*np.sin(phi2)
    pz1 = np.abs(pt1)/np.tan(2.0*np.arctan(np.exp(-1.*eta1)))
    pz2 = np.abs(pt2)/np.tan(2.0*np.arctan(np.exp(-1.*eta2)))
    return np.c_[e1,px1,py1,pz1,e2,px2,py2,pz2]

def get_dphi(px1,py1,px2,py2):
    phi1 = np.arctan2(py1,px1)
    phi2 = np.arctan2(py2,px2)
    dphi = phi1-phi2
    dphi[dphi>np.pi] -= 2*np.pi
    dphi[dphi<-np.pi] += 2*np.pi 
    return dphi

def get_rotated_pxpy(px1,py1,px2,py2):
    """
    rotates two leptons such that phi2 = 0
    """
    pt1 = np.sqrt(px1**2+py1**2)
    pt2 = np.sqrt(px2**2+py2**2)
    phi1 = np.arctan2(py1,px1)
    phi2 = np.arctan2(py2,px2)
    px1p = pt1*np.cos(phi1-phi2)
    py1p = pt1*np.sin(phi1-phi2)
    px2p = pt2*np.cos(phi2-phi2)
    return px1p,py1p,px2p,np.zeros(len(pt2))
    
def cartesian_zerophi2(coords,ptetaphi=False):
    """
    returns 8-1=7 columns rotating leptons such that phi2 is 0 (and removing it)
    if `ptetaphi` is True, then return eptetaphi instead of epxpypz
    """
    lepcoords_cyl = cartesian_to_ptetaphi(coords)
    phi1 = lepcoords_cyl[:,3]
    phi2 = lepcoords_cyl[:,7]
    dphi = phi1-phi2
    dphi[dphi>np.pi] -= 2*np.pi
    dphi[dphi<-np.pi] += 2*np.pi
    lepcoords_cyl[:,3] = dphi
    lepcoords_cyl[:,7] = 0.
    if ptetaphi:
        return np.delete(lepcoords_cyl, [7], axis=1)
    else:
        return np.delete(ptetaphi_to_cartesian(lepcoords_cyl), [6], axis=1)

def invmass_from_8cartesian(x):
    invmass = K.sqrt(
                (x[:,0:1]+x[:,4:5])**2-
                (x[:,1:2]+x[:,5:6])**2-
                (x[:,2:3]+x[:,6:7])**2-
                (x[:,3:4]+x[:,7:8])**2
                )
    return invmass

def invmass_from_8cartesian_nopy2(x):
    invmass = K.sqrt(
                (x[:,0:1]+x[:,4:5])**2-
                (x[:,1:2]+x[:,5:6])**2-
                (x[:,2:3]         )**2-
                (x[:,3:4]+x[:,6:7])**2
                )
    return invmass

def get_first_N(x,N=19):
    return x[:,0:N]

def add_invmass_from_8cartesian(x):
    return K.concatenate([x,invmass_from_8cartesian(x)])


def fix_outputs(x):
    """
    Take nominal delphes format of 19 columns and fix some columns
    """
    return K.concatenate([
        # x[:,0:21],
        x[:,0:7], # epxpypz for lep1,lep2 -1 for no py2
        x[:,7:8], # nvtx
        K.sign(x[:,8:10]), # q1 q2
        x[:,10:12], # iso1 iso2
        x[:,12:14], # met, metphi
        x[:,14:19], # jet pts
        ])

def custom_loss(c, loss_type = "force_mll"):
    if loss_type == "force_mll":
        def loss_func(y_true, y_pred_mll):
            y_true = y_true[:,0]
            y_pred = y_pred_mll[:,0]
            mll_pred = y_pred_mll[:,1]

            mll_loss = K.mean(K.abs(mll_pred - 91.2))

    #         pseudomll = K.random_normal_variable(shape=(1,1), mean=91.2, scale=2)
    #         mll_loss = K.mean((mll_pred - pseudomll)**2)

            return binary_crossentropy(y_true, y_pred) + c*mll_loss
        return loss_func
    elif loss_type == "force_z_width":
        def loss_func(y_true, y_pred_mll):
            y_true = y_true[:,0]
            y_pred = y_pred_mll[:,0]
            mll_pred = y_pred_mll[:,1]
            
            mll_loss = K.mean(K.abs(mll_pred - 91.2))
            mll_sigma_loss = K.abs(K.std(mll_pred) - 7.67)

            return binary_crossentropy(y_true, y_pred) + c*mll_loss + c*mll_sigma_loss
        return loss_func
        
    else:
        raise ValueError("Can not make loss function of type %s" % loss_type)

def METPhiMap(metphis):
    """Works so long as the tails are constrained within [-2pi, 2pi], maps everything from [-pi,pi]"""
    return ((metphis+np.pi) % (2*np.pi)) - np.pi

def make_plots_new(preds,reals,title="",fname="",show_pred=True,wspace=0.1,hspace=0.3,tightlayout=True,visible=False):
    nrows, ncols = 5,5
    fig, axs = plt.subplots(nrows,ncols,figsize=(16,13))
#     fig, axs = plt.subplots(nrows,ncols,figsize=(12,10))
#     fig.subplots_adjust(wspace=0.1,hspace=0.3)
    fig.subplots_adjust(wspace=wspace,hspace=hspace)


    info = [
        ["lep1_e",(0,250,50)],
        ["lep1_px",(-100,100,50)],
        ["lep1_py",(-100,100,50)],
        ["lep1_pz",(-200,200,50)],
        ["lep2_e",(0,250,50)],
        ["lep2_px",(-100,100,50)],
        ["lep2_pz",(-200,200,50)],
        ["dphi",(-4,4,50)],
        ["nvtxs",(0,50,350)],
        ["met",(0,150,50)],
        ["metphi",(-6,6,50)],
        ["lep1_charge",(-7,7,30)],
        ["lep2_charge",(-7,7,30)],
        ["lep1_iso",(0,2.0,30)],
        ["lep2_iso",(0,2.0,30)],
        ["jet_pt1",(0,100,50)],
        ["jet_pt2",(0,100,50)],
        ["jet_pt3",(0,100,50)],
        ["jet_pt4",(0,100,50)],
        ["jet_pt5",(0,100,50)],
        ["mll",(60,120,50)],
        ["lep1_mass",(0,1,50)],
        ["lep2_mass",(0,1,50)],
        ["njets",(0,7,7)],

    ]
    for axx in axs:
        for ax in axx:
            ax.get_yaxis().set_visible(False)
    for ic,(cname,crange) in enumerate(info):
        if cname == "mll":
            real = reals["mll"]
            pred = Minv(preds,ptetaphi=False,nopy2=True)
        elif cname == "lep1_mass": real, pred = M4(reals["lep1_e"], reals["lep1_px"], reals["lep1_py"], reals["lep1_pz"]), M4(preds[:,0], preds[:,1], preds[:,2], preds[:,3])
        elif cname == "lep2_mass": real, pred = M4(reals["lep2_e"], reals["lep2_px"], 0, reals["lep2_pz"]), M4(preds[:,4], preds[:,5], preds[:,6], preds[:,7])
        elif cname == "lep1_e": real, pred = reals[cname], preds[:,0]
        elif cname == "lep1_pz": real, pred = reals[cname], preds[:,3]
        elif cname == "lep2_e": real, pred = reals[cname], preds[:,4]
        elif cname == "lep2_pz": real, pred = reals[cname], preds[:,6]
        elif cname == "lep1_px": 
            real = reals[cname]
            pred = preds[:,1]
        elif cname == "lep1_py":
            real = reals[cname]
            pred = preds[:,2]
        elif cname == "lep2_px":
            real = reals[cname]
            pred = preds[:,5]
        elif cname == "dphi":
            real = get_dphi(reals["lep1_px"], reals["lep1_py"], reals["lep2_px"], np.zeros(len(reals)))
            pred = get_dphi(preds[:,1], preds[:,2], preds[:,5], np.zeros(len(preds)))
        elif cname == "nvtxs": real, pred = reals[cname], np.round(preds[:,7])
        elif cname == "lep1_charge": real, pred = reals[cname], preds[:,8]
        elif cname == "lep2_charge": real, pred = reals[cname], preds[:,9]
        elif cname == "lep1_iso": real, pred = reals[cname], preds[:,10]
        elif cname == "lep2_iso": real, pred = reals[cname], preds[:,11]
        elif cname == "met": real, pred = reals[cname], preds[:,12]
        elif cname == "metphi": real, pred = reals[cname], METPhiMap(preds[:,13])
        elif cname == "jet_pt1": real, pred = reals[cname], preds[:,14]
        elif cname == "jet_pt2": real, pred = reals[cname], preds[:,15]
        elif cname == "jet_pt3": real, pred = reals[cname], preds[:,16]
        elif cname == "jet_pt4": real, pred = reals[cname], preds[:,17]
        elif cname == "jet_pt5": real, pred = reals[cname], preds[:,18]
        elif cname == "njets":
            real = \
                1*(reals["jet_pt1"] > 10) + \
                1*(reals["jet_pt2"] > 10) + \
                1*(reals["jet_pt3"] > 10) + \
                1*(reals["jet_pt4"] > 10) + \
                1*(reals["jet_pt5"] > 10)
            pred = \
                1*(preds[:,14] > 10) + \
                1*(preds[:,15] > 10) + \
                1*(preds[:,16] > 10) + \
                1*(preds[:,17] > 10) + \
                1*(preds[:,18] > 10)
        idx = ic // ncols, ic % ncols
        bins_real = axs[idx].hist(real, range=crange[:2],bins=crange[-1], histtype="step", lw=1.5,density=True)
        if show_pred:
            bins_pred = axs[idx].hist(pred, range=crange[:2],bins=crange[-1], histtype="step", lw=1.5,density=True)
        axs[idx].set_xlabel("{}".format(cname))
        axs[idx].get_yaxis().set_visible(False)
    #     axs[idx].set_yscale("log", nonposy='clip')
    _ = axs[0,0].legend(["True","Pred"], loc='upper right')
    _ = axs[0,0].set_title(title)
    if tightlayout:
        plt.tight_layout()
    if fname:
        fig.savefig(fname)
    if not visible:
        plt.close(fig)

def make_plots_old(preds,reals,title="",fname="", visible=False):
    nrows, ncols = 5,5
    fig, axs = plt.subplots(nrows,ncols,figsize=(16,13))
    fig.subplots_adjust(wspace=0.1,hspace=0.3)


    #print(preds)
    info = [
        ["lep1_e",(0,250,50)],
        ["lep1_px",(-100,100,50)],
        ["lep1_py",(-100,100,50)],
        ["lep1_pz",(-200,200,50)],
        ["lep2_e",(0,250,50)],
        ["lep2_px",(-100,100,50)],
        ["lep2_pz",(-200,200,50)],
        ["dphi",(-4,4,50)],
        ["nvtxs",(0,50,350)],
        ["met",(0,150,50)],
        ["metphi",(-6,6,50)],
        ["lep1_charge",(-7,7,30)],
        ["lep2_charge",(-7,7,30)],
        ["lep1_iso",(0,0.2,30)],
        ["lep2_iso",(0,0.2,30)],
        ["genjet_pt1",(0,100,50)],
        ["genjet_pt2",(0,100,50)],
        ["genjet_pt3",(0,100,50)],
        ["genjet_pt4",(0,100,50)],
        ["genjet_pt5",(0,100,50)],
        ["mll",(60,120,50)],
        ["lep1_mass",(0,1,50)],
        ["lep2_mass",(0,1,50)],
        ["njets",(0,7,7)],

    ]
    for ic,(cname,crange) in enumerate(info):
        if cname == "mll":
            real = reals["mll"]
            pred = Minv(preds,ptetaphi=False,nopy2=True)
        elif cname == "lep1_mass": real, pred = M4(reals["lep1_e"], reals["lep1_px"], reals["lep1_py"], reals["lep1_pz"]), M4(preds[:,0], preds[:,1], preds[:,2], preds[:,3])
        elif cname == "lep2_mass": real, pred = M4(reals["lep2_e"], reals["lep2_px"], 0, reals["lep2_pz"]), M4(preds[:,4], preds[:,5], preds[:,6], preds[:,7])
        elif cname == "lep1_e": real, pred = reals[cname], preds[:,0]
        elif cname == "lep1_pz": real, pred = reals[cname], preds[:,3]
        elif cname == "lep2_e": real, pred = reals[cname], preds[:,4]
        elif cname == "lep2_pz": real, pred = reals[cname], preds[:,6]
        elif cname == "lep1_px": 
            real = get_rotated_pxpy(reals["lep1_px"], reals["lep1_py"], reals["lep2_px"], reals["lep2_py"])[0]
            pred = preds[:,1]
        elif cname == "lep1_py":
            real = get_rotated_pxpy(reals["lep1_px"], reals["lep1_py"], reals["lep2_px"], reals["lep2_py"])[1]
            pred = preds[:,2]
        elif cname == "lep2_px":
            real = get_rotated_pxpy(reals["lep1_px"], reals["lep1_py"], reals["lep2_px"], reals["lep2_py"])[2]
            pred = preds[:,5]
        elif cname == "dphi":
            real = get_dphi(reals["lep1_px"], reals["lep1_py"], reals["lep2_px"], reals["lep2_py"])
            pred = get_dphi(preds[:,1], preds[:,2], preds[:,5], np.zeros(len(preds)))
        elif cname == "nvtxs": real, pred = reals[cname], np.round(preds[:,7])
        elif cname == "lep1_charge": real, pred = reals[cname], preds[:,8]
        elif cname == "lep2_charge": real, pred = reals[cname], preds[:,9]
        elif cname == "lep1_iso": real, pred = reals[cname], preds[:,10]
        elif cname == "lep2_iso": real, pred = reals[cname], preds[:,11]
        elif cname == "met": real, pred = reals[cname], preds[:,12]
        elif cname == "metphi": real, pred = reals[cname], METPhiMap(preds[:,13])
        elif cname == "genjet_pt1": real, pred = reals[cname], preds[:,14]
        elif cname == "genjet_pt2": real, pred = reals[cname], preds[:,15]
        elif cname == "genjet_pt3": real, pred = reals[cname], preds[:,16]
        elif cname == "genjet_pt4": real, pred = reals[cname], preds[:,17]
        elif cname == "genjet_pt5": real, pred = reals[cname], preds[:,18]
        idx = ic // ncols, ic % ncols
        bins_real = axs[idx].hist(real, range=crange[:2],bins=crange[-1], histtype="step", lw=2,density=True)
        bins_pred = axs[idx].hist(pred, range=crange[:2],bins=crange[-1], histtype="step", lw=2,density=True)
        axs[idx].set_xlabel("{}".format(cname))
        axs[idx].get_yaxis().set_visible(False)
    #     axs[idx].set_yscale("log", nonposy='clip')
    _ = axs[0,0].legend(["True","Pred"], loc='upper right')
    _ = axs[0,0].set_title(title)
    plt.tight_layout()
    if fname:
        fig.savefig(fname)
    if not visible:
        plt.close(fig)

make_plots = None #will be set by get_noise method



def load_data(input_file):
    data = np.load(input_file)
    return data[data["genmll"] > 50.]

def get_noise(input_file, data, noise_type, noise_shape, amount=1024, use_ptetaphi_additionally=False, use_delphes=True):
    # nominal
    global make_plots
    
    if "data_Nov10.npa" in input_file:
        make_plots = make_plots_old
        
        lepcoords = np.c_[
            data["lep1_e"],
            data["lep1_px"],
            data["lep1_py"],
            data["lep1_pz"],
            data["lep2_e"],
            data["lep2_px"],
            data["lep2_py"],
            data["lep2_pz"],
        ]
        lepcoords_dphi = cartesian_zerophi2(lepcoords)

        nvtx_smeared = np.round(np.random.normal(data["nvtxs"],0.5))
        X_train = np.c_[
            lepcoords_dphi, # 7 columns
            nvtx_smeared, # 1 column
            data["lep1_charge"], data["lep2_charge"],
            data["lep1_iso"], data["lep2_iso"],
            data["met"], data["metphi"],
            data["genjet_pt1"],
            data["genjet_pt2"],
            data["genjet_pt3"],
            data["genjet_pt4"],
            data["genjet_pt5"],
        ].astype(np.float32)

    elif "total_Zmumu_13TeV_PU20_v2.npa" in input_file:
        make_plots = make_plots_new
        lepcoords = np.c_[
            data["lep1_e"],
            data["lep1_px"],
            data["lep1_py"],
            data["lep1_pz"],
            data["lep2_e"],
            data["lep2_px"],
#                 data["lep2_py"],
            data["lep2_pz"],
        ]
#             lepcoords_dphi = cartesian_zerophi2(lepcoords)

        nvtx_smeared = np.round(np.random.normal(data["nvtxs"],0.5))
        X_train = np.c_[
#                 lepcoords_dphi, # 7 columns
            lepcoords, # 7 columns
            nvtx_smeared, # 1 column
            data["lep1_charge"], data["lep2_charge"],
            data["lep1_iso"], data["lep2_iso"],
            data["met"], data["metphi"],
            data["jet_pt1"],
            data["jet_pt2"],
            data["jet_pt3"],
            data["jet_pt4"],
            data["jet_pt5"],
        ].astype(np.float32)
    else:
        make_plots = make_plots_old
        X_train = data[:,range(1,1+8)]
        if use_ptetaphi_additionally:
            X_train = np.c_[X_train, cartesian_to_ptetaphi(X_train)]

            
    if noise_type == 1:
        noise_half = np.random.normal(0, 1, (amount//2, noise_shape[0]))
        noise_full = np.random.normal(0, 1, (amount, noise_shape[0]))

    elif noise_type == 2: # random soup, 4,2,2 have to be modified to sum to noise_shape[0]
        ngaus = noise_shape[0] // 2
        nflat = (noise_shape[0] - ngaus) // 2
        nexpo = noise_shape[0] - nflat - ngaus
        noise_gaus = np.random.normal( 0, 1, (amount//2+amount, ngaus))
        noise_flat = np.random.uniform(-1, 1, (amount//2+amount, nflat))
        noise_expo = np.random.exponential( 1,    (amount//2+amount, nexpo))
        noise = np.c_[ noise_gaus,noise_flat,noise_expo ]
        noise_half = noise[:amount//2]
        noise_full = noise[-amount:]

    elif noise_type == 3: # truth conditioned

#             noise_half = np.c_[ 
#                     self.X_train[np.random.randint(0, self.X_train.shape[0], amount//2)], 
#                     np.random.normal(0, 1, (amount//2,self.noise_shape[0]-self.X_train.shape[1]))
#                     ]
#             noise_full = np.c_[ 
#                     self.X_train[np.random.randint(0, self.X_train.shape[0], amount)], 
#                     np.random.normal(0, 1, (amount,self.noise_shape[0]-self.X_train.shape[1]))
#                     ]

        npurenoise = noise_shape[0]-X_train.shape[1]
        ngaus = npurenoise // 2
        nflat = (npurenoise - ngaus) // 2
        nexpo = npurenoise - nflat - ngaus
        noise_gaus = np.random.normal( 0, 1, (amount//2+amount, ngaus))
        noise_flat = np.random.uniform(-1, 1, (amount//2+amount, nflat))
        noise_expo = np.random.exponential( 1,    (amount//2+amount, nexpo))
        truenoise = X_train[np.random.randint(0, X_train.shape[0], amount//2+amount)]
        noise = np.c_[ truenoise,noise_gaus,noise_flat,noise_expo ]
        noise_half = noise[:amount//2]
        noise_full = noise[-amount:]

    return X_train, noise_half, noise_full



def onetime(func):
    """stores the functions output, returns the output if called again on the same input, else computes new output"""
    def decorated(*args, **kwargs):
        global cov_ans
        global cov_hash
        new_hash=hashlib.md5(str(args)+str(kwargs)).hexdigest() 
        if new_hash != cov_hash:
            #print("computing")
            cov_ans = func(*args, **kwargs)
        cov_hash=new_hash
        return cov_ans
    return decorated
    

@onetime
def covariance_metrics(real_data, predictions):
    """Takes in real_data matrix with real entries as rows and predictions matrix with generated events as rows and returns the covariance matricies for the two as well as the average, maximum, and std. dev of the difference between the entries in the coverance matrix as well as in the average of the variables."""
    
    cov_pred = np.cov(predictions.T)
    avg_pred = predictions.mean(axis=0)
    cov_real = np.cov(real_data.T)
    avg_real = real_data.mean(axis=0)
    
    #cov_diff = np.abs((cov_pred - cov_real)/np.sqrt(np.abs(np.outer(avg_real, avg_pred))))
    cov_diff = np.abs((cov_pred - cov_real)/cov_real)
    ar=avg_real
    ar[ar == 0] = 1
    avg_diff = np.abs((avg_pred - avg_real)/ar)
    
    return cov_diff, avg_diff

def get_score(real_data, predictions, weight_cov = (1/361.), weight_avg = (1/19.)):
    cov_diff, avg_diff = covariance_metrics(real_data, predictions)
    return weight_cov*np.sum(cov_diff)+weight_avg*np.sum(avg_diff)

def getKS(real_data, predictions):
    return ks_2samp(Minv(real_data,ptetaphi=False,nopy2=True), Minv(predictions,ptetaphi=False,nopy2=True))


tag = "v6_batch512_bgbd_mllANDwidth_NonTC_newdata_mllfix"
epoch=53000
trial=12
input_file = "/home/users/bhashemi/Projects/GIT/DY-GAN/delphes/total_Zmumu_13TeV_PU20_v2.npa"

data = load_data(input_file)

loss_type = "force_z_width"
loss_weight = 0.01

#noise_shape = (8,)
noise_shape = (19,)
event_count=50000

#f=open("progress/%s/newlog.txt" % tag, "w+")
print("about to run on %s" % tag)
model = load_model("progress/%s/gen_%d.weights" % (tag,epoch), custom_objects={'loss_func': custom_loss(c=loss_weight, loss_type=loss_type)})

Stat_scores = []
MLL_scores = []
MetPhi_scores = []
LepIso_scores = []

for i in xrange(1,100):
    real_events, noise_half, noise = get_noise(input_file, data, 1, noise_shape, event_count)
    real = real_events[np.random.randint(0,real_events.shape[0], event_count)]
    preds = model.predict(noise)
    score = get_score(real, preds)
    mll_ks_score = ks_2samp(Minv(real,ptetaphi=False,nopy2=True), Minv(preds,ptetaphi=False,nopy2=True))
    MetPhi_ks_score = ks_2samp(METPhiMap(real[:,13]), METPhiMap(preds[:,13]))
    Lep1Iso_ks_score = ks_2samp(real[:,10], preds[:,10])
    print "%d: epoch %d trial %d StatsScore %f MLLKSStatistic %f MLLKSPval %f METPhiKSStatistic %f METPhiKSPval %f Lep1IsoKSStatistic %f Lep1IsoKSPval %f " % (i, epoch, trial, score, mll_ks_score[0], mll_ks_score[1], MetPhi_ks_score[0], MetPhi_ks_score[1], Lep1Iso_ks_score[0], Lep1Iso_ks_score[1])
    
    Stat_scores.append(score)
    MLL_scores.append(mll_ks_score[0])
    MetPhi_scores.append(MetPhi_ks_score[0])
    LepIso_scores.append(Lep1Iso_ks_score[0])

print "=================="
print "Stat: Max %f Min %f Delta %f Avg %f StdDev %f" % (np.max(Stat_scores), np.min(Stat_scores), np.max(Stat_scores)-np.min(Stat_scores), np.mean(Stat_scores), np.std(Stat_scores))
print "MLL: Max %f Min %f Delta %f Avg %f StdDev %f" % (np.max(MLL_scores), np.min(MLL_scores), np.max(MLL_scores)-np.min(MLL_scores), np.mean(MLL_scores), np.std(MLL_scores))
print "MetPhi: Max %f Min %f Delta %f Avg %f StdDev %f" % (np.max(MetPhi_scores), np.min(MetPhi_scores), np.max(MetPhi_scores)-np.min(MetPhi_scores), np.mean(MetPhi_scores), np.std(MetPhi_scores))
print "LepIso: Max %f Min %f Delta %f Avg %f StdDev %f" % (np.max(LepIso_scores), np.min(LepIso_scores), np.max(LepIso_scores)-np.min(LepIso_scores), np.mean(LepIso_scores), np.std(LepIso_scores))

"""for trial in xrange(4,15):
    tag = "v%d_batch512_bgbd_mllANDwidth_NonTC_newdata_mllfix" % trial

    #Old file with Gen/Reco mix
    #input_file = "/home/users/bhashemi/Projects/GIT/DY-GAN/delphes/data_Nov10.npa"
    #New Files with Gen and Reco split
    input_file = "/home/users/bhashemi/Projects/GIT/DY-GAN/delphes/total_Zmumu_13TeV_PU20_v2.npa"

    data = load_data(input_file)
    #loss_type = "force_mll",
    loss_type = "force_z_width"
    loss_weight = 0.01

    #noise_shape = (8,)
    noise_shape = (19,)

    f=open("progress/%s/newlog.txt" % tag, "w+")
    print("about to run on %s" % tag)

    for epoch in xrange(500,100001,500):
        try:
            model = load_model("progress/%s/gen_%d.weights" % (tag,epoch), custom_objects={'loss_func': custom_loss(c=loss_weight, loss_type=loss_type)})
        except Exception as e:
            continue
        real_events, noise_half, noise = get_noise(input_file, data, 1, noise_shape, 50000)
        real = real_events[:50000]
        preds = model.predict(noise)
        score = get_score(real, preds)
        mll_ks_score = ks_2samp(Minv(real,ptetaphi=False,nopy2=True), Minv(preds,ptetaphi=False,nopy2=True))
        MetPhi_ks_score = ks_2samp(METPhiMap(real[:,13]), METPhiMap(preds[:,13]))
        Lep1Iso_ks_score = ks_2samp(real[:,10], preds[:,10])
        print "epoch %d trial %d StatsScore %f MLLKSStatistic %f MLLKSPval %f METPhiKSStatistic %f METPhiKSPval %f Lep1IsoKSStatistic %f Lep1IsoKSPval %f " % (epoch, trial, score, mll_ks_score[0], mll_ks_score[1], MetPhi_ks_score[0], MetPhi_ks_score[1], Lep1Iso_ks_score[0], Lep1Iso_ks_score[1])
        f.write("epoch %d trial %d StatsScore %f MLLKSStatistic %f MLLKSPval %f METPhiKSStatistic %f METPhiKSPval %f Lep1IsoKSStatistic %f Lep1IsoKSPval %f\n" % (epoch, trial, score, mll_ks_score[0], mll_ks_score[1], MetPhi_ks_score[0], MetPhi_ks_score[1], Lep1Iso_ks_score[0], Lep1Iso_ks_score[1]))
        
    f.close()"""