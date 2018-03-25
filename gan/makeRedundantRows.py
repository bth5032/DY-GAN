import numpy as np
import physicsfuncs as pf

data = np.loadtxt(open("short.input", "r"), delimiter=",", skiprows=1)
cart_cols = data[:,1:1+8]
to_append = pf.getLepsPtThetaPhi(cart_cols)
