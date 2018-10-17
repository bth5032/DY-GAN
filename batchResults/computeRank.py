#!/usr/bin/python
import os

use_ss = use_mll = use_metphi = use_lepiso = False

def rankKey(info):
  global use_ss, use_mll, use_metphi, use_lepiso
  score=0
  if use_ss:
    score+=info[1][3]
  if use_mll:
    score+=info[1][2]
  if use_metphi:
    score+=info[1][1]
  if use_lepiso:
    score+=info[1][0]
  return score


fnames = ["score.txt", "MLL.txt", "MetPhi.txt", "LepIso.txt"]

ranks = {}
once=False
for name in fnames:
  rank=0
  with open("logs/%s" % name, "r") as f:
    for line in f:
      toks = line.split()
      score = toks[5]
      mll = toks[7]
      metphi = toks[11]
      lepiso = toks[15]
      trial = toks[3]
      epoch = toks[1]
      if (score == "nan"):
        pass
      else:
        rank+=1
        try:
          ranks[(trial, epoch)].insert(0, rank)
        except:
          ranks[(trial, epoch)] = [rank, score, mll, metphi, lepiso]


tests = {"lepiso": (False, False, False, True),
         "metphi": (False, False, True, False),
         "mll": (False, True, False, False),
         "ss": (True, False, False, False),
         "ss_mll": (True, True, False, False),
         "ss_mll_metphi": (True, True, True, False),
         "mll_metphi": (False, True, True, False),
         "mll_metphi_lepiso": (False, True, True, True),
         "mll_lepiso": (False, True, False, True),
         "metphi_lepiso": (False, False, True, True),
         "all": (True, True, True, True),
         }

for name, args in tests.items():
  use_ss, use_mll, use_metphi, use_lepiso = args
  best=sorted(ranks.items(), key=rankKey)
  fname = name+"_rankings.txt"
  f=open(fname, "w+")
  f.write("Trial\tEpoch\tLepIsoRank\tMetPhiRank\tMLLRank\tScoreRank\tStatsScore\tMLLKS\tMetPhiKS\tLepIsoKS\tSortKey\n")
  for i in best:
    f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%f\n" % (i[0][0], i[0][1], i[1][0], i[1][1], i[1][2], i[1][3], i[1][4], i[1][5], i[1][6], i[1][7], rankKey(i)))
  f.close()
  os.system("cat %s | column -t > results/%s" % (fname, fname))
  os.system("rm %s" % (fname))