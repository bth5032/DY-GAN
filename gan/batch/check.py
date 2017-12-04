from tqdm import tqdm
import os
import sys
from glob import glob
import numpy as np
import pickle


dirs = glob("/hadoop/cms/store/user/namin/gan_output/v3/*/")
total = 0
d_total = {}
for thedir in tqdm(dirs):
    fname = "{}/history.pkl".format(thedir)
    if not os.path.exists(fname): continue
    num = int(thedir.rstrip("/").rsplit("_",1)[-1])

    total += 1
    data = pickle.load(open(fname,"r"))
    d_total[num] = data

print "Found {} total pickle files".format(total)
pickle.dump(d_total, open("total_info.pkl","w"))
