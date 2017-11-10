#!/usr/bin/env python

import sys
import os
import commands
import json
import pickle
import time

def update(fname):
    with open("web/data.js","w") as fhout:
        fhout.write("info = {};".format(pickle.load(open(fname,"r"))))
        print "Updated dashboard"
    os.system("cp web/data.js ~/public_html/dump/")
    os.system("cp web/view.html ~/public_html/dump/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("You didn't give me a tag!")
        sys.exit()
    tag = sys.argv[-1].strip()

    # tag = "vnonoise"

    fname = "progress/{}/history.pkl".format(tag)

    if not os.path.exists(fname):
        raise Exception("{} doesn't exist!".format(fname))

    print "Monitoring {}...".format(fname)

    while True:
        update(fname)
        time.sleep(20)
