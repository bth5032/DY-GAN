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