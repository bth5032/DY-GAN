import math

def M_ptetaphi(mom):
  """Outputs the Mass of a two particle system when given a list of E1, pt1, eta1, phi1, E2, pt2, eta2, phi2"""
  E1, pt1, eta1, phi1, E2, pt2, eta2, phi2 = map(float, mom)
  p1x = pt1*math.cos(phi1)
  p1y = pt1*math.sin(phi1)
  p1z = pt1*math.sinh(eta1)

  p2x = pt2*math.cos(phi2)
  p2y = pt2*math.sin(phi2)
  p2z = pt2*math.sinh(eta2)

  return M(E1, px1, py1, pz1, E2, px2, py2, pz2)

def M_xyz(mom):
  """Outputs the Mass of a two particle system when given a list of E1, px1, py1, pz1, E2, px2, py2, pz2"""
  return M(*map(float, mom))

def M(E1, px1, py1, pz1, E2, px2, py2, pz2):
  """Computes M for two objects given the Carteasan momentum projections"""
  M = (E1+E2)**2 - (p1x+p2x)**2 - (p1y+p2y)**2 - (p1z+p2z)**2
  return (str(math.sqrt(M)) if (M > 0) else None)