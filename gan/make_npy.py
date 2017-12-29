import os
import sys
import argparse
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="name of input text file")
    # parser.add_argument("-c", "--compress", help="compress to npz instead of npy", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        raise RuntimeError("File {} doesn't exist, dummy.".format(args.filename))

    data = np.loadtxt(args.filename, delimiter=",").astype(np.float32)
    outname = args.filename.replace(".txt","$$").replace(".csv","$$").replace("$$",".npy")
    print ">>> Saving to {}".format(outname)
    data.dump(outname)
    # np.savez_compressed(outname.replace(".npy",".npz"), data=data)
