import numpy as np
from scipy.stats import moment

def M_ptetaphi(E1, pt1, eta1, phi1, E2, pt2, eta2, phi2):
  """Outputs the Mass of a two particle system when given a list of E1, pt1, eta1, phi1, E2, pt2, eta2, phi2"""
  p1x = pt1*np.cos(phi1)
  p1y = pt1*np.sin(phi1)
  p1z = pt1*np.sinh(eta1)

  p2x = pt2*np.cos(phi2)
  p2y = pt2*np.sin(phi2)
  p2z = pt2*np.sinh(eta2)

  return M(E1, p1x, p1y, p1z, E2, p2x, p2y, p2z)

def M(E1, p1x, p1y, p1z, E2, p2x, p2y, p2z):
  """Computes M for two objects given the Carteasan momentum projections"""
  M = (E1+E2)**2 - (p1x+p2x)**2 - (p1y+p2y)**2 - (p1z+p2z)**2
  return np.sqrt(M)

def Minv(eight_cartesian_cols):
  """Computes M for two objects given the Carteasan momentum projections"""
  M2 = (eight_cartesian_cols[:,0]+eight_cartesian_cols[:,4])**2
  M2 -= (eight_cartesian_cols[:,1]+eight_cartesian_cols[:,5])**2
  M2 -= (eight_cartesian_cols[:,2]+eight_cartesian_cols[:,6])**2
  M2 -= (eight_cartesian_cols[:,3]+eight_cartesian_cols[:,7])**2
  return np.sqrt(M2)


def get_pxs(eight_cartesian_cols):
    return np.r_[ eight_cartesian_cols[:,1], eight_cartesian_cols[:,5] ]

def get_pys(eight_cartesian_cols):
    return np.r_[ eight_cartesian_cols[:,2], eight_cartesian_cols[:,6] ]

def get_pts(eight_cartesian_cols):
    px1 = eight_cartesian_cols[:,1]
    py1 = eight_cartesian_cols[:,2]
    px2 = eight_cartesian_cols[:,5]
    py2 = eight_cartesian_cols[:,6]
    return np.r_[ np.sqrt(px1**2+py1**2), np.sqrt(px2**2+py2**2) ]

def cartesian_to_ptetaphi(eight_cartesian_cols):
    """
    Takes 8 columns as cartesian
    e px py pz e px py pz
    and converts to
    e pt eta phi e pt eta phi
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
    and converts to
    e px py pz e px py pz
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

def get_phis(eight_cartesian_cols):
    """Get vector of phis for both leptons"""
    px1 = eight_cartesian_cols[:,1]
    px2 = eight_cartesian_cols[:,5]
    py1 = eight_cartesian_cols[:,2]
    py2 = eight_cartesian_cols[:,6]
    return  np.r_[np.arctan2(py1,px1),np.arctan2(py2,px2)]

def get_dphis(eight_cartesian_cols):
    """Get vector of dphis between both leptons"""
    px1 = eight_cartesian_cols[:,1]
    px2 = eight_cartesian_cols[:,5]
    py1 = eight_cartesian_cols[:,2]
    py2 = eight_cartesian_cols[:,6]
    dphis = np.abs(np.arctan2(py1,px1)-np.arctan2(py2,px2))
    dphis[dphis > 3.14159] -= 3.141592
    return dphis

def get_etas(eight_cartesian_cols):
    """Get vector of etas for both leptons
    https://en.wikipedia.org/wiki/Pseudorapidity"""
    px1 = eight_cartesian_cols[:,1]
    px2 = eight_cartesian_cols[:,5]
    py1 = eight_cartesian_cols[:,2]
    py2 = eight_cartesian_cols[:,6]
    pz1 = eight_cartesian_cols[:,3]
    pz2 = eight_cartesian_cols[:,7]
    p1 = np.sqrt(px1**2+py1**2+pz1**2)
    p2 = np.sqrt(px2**2+py2**2+pz2**2)
    return  np.r_[np.arctanh(pz1/p1),np.arctanh(pz2/p2)]

def get_detas(eight_cartesian_cols):
    """Get vector of delta etas for both leptons
    https://en.wikipedia.org/wiki/Pseudorapidity"""
    px1 = eight_cartesian_cols[:,1]
    px2 = eight_cartesian_cols[:,5]
    py1 = eight_cartesian_cols[:,2]
    py2 = eight_cartesian_cols[:,6]
    pz1 = eight_cartesian_cols[:,3]
    pz2 = eight_cartesian_cols[:,7]
    p1 = np.sqrt(px1**2+py1**2+pz1**2)
    p2 = np.sqrt(px2**2+py2**2+pz2**2)
    return  np.abs(np.arctanh(pz1/p1)-np.arctanh(pz2/p2))

def Z_pZ(eight_cartesian_cols):
    return eight_cartesian_cols[:,3]+eight_cartesian_cols[:,7]

def Z_pT(eight_cartesian_cols):
    pxsum = eight_cartesian_cols[:,1]+eight_cartesian_cols[:,5]
    pysum = eight_cartesian_cols[:,2]+eight_cartesian_cols[:,6]
    return np.sqrt(pxsum**2+pysum**2)
    

def get_momentmetric(obs,exp):
    """
    return a metric of agreement based on the moments of the input distributions
    """
    obs[~np.isfinite(obs)] = 0
    return np.abs(moment(obs,2)-moment(exp,2))/1e6 + \
           np.abs(moment(obs,3)-moment(exp,3))/1e7 + \
           np.abs(moment(obs,4)-moment(exp,4))/1e8

def get_redchi2(obs,exp):
    """
    return reduced chi2 (chi2/ndof) for observed and expected values.
    both inputs must have the same length (i.e., their integral must
    be the same, as no normalization is done here).
    """
    obs[~np.isfinite(obs)] = 0
    counts1, bins1 = np.histogram(obs)
    counts2, bins2 = np.histogram(exp, bins1)
    tosum = (1.0*(counts1-counts2)**2 / counts2)
    return np.sum(tosum[np.isfinite(tosum)])/np.sum(np.isfinite(tosum))

def get_quantities(data):
    return {
            "mll": Minv(data),
            "ZpZ": Z_pZ(data),
            "ZpT": Z_pT(data),
            "px": get_pxs(data),
            "py": get_pys(data),
            "dphi": get_dphis(data),
            "pt": get_pts(data),
            "eta": get_etas(data),
            "deta": get_detas(data),
            }

def get_metrics(data_obs, data_exp):
    d_obs = get_quantities(data_obs)
    d_exp = get_quantities(data_exp)
    metric1, metric2 = 0., 0.
    for key in d_obs:
        metric1 += get_momentmetric(d_obs[key], d_exp[key])
        metric2 += get_redchi2(d_obs[key], d_exp[key])
    metric1 /= len(d_obs.keys())
    metric2 /= len(d_obs.keys())
    metric2 = np.log(metric2)
    return metric1, metric2
