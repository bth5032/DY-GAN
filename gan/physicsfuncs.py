import numpy as np

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


def Z_pZ(eight_cartesian_cols):
    return eight_cartesian_cols[:,3]+eight_cartesian_cols[:,7]

def Z_pT(eight_cartesian_cols):
    pxsum = eight_cartesian_cols[:,1]+eight_cartesian_cols[:,5]
    pysum = eight_cartesian_cols[:,2]+eight_cartesian_cols[:,6]
    return np.sqrt(pxsum**2+pysum**2)
    
