import math

def M(mom):
  E1, pt1, eta1, phi1, E2, pt2, eta2, phi2 = map(float, mom)
  p1x = pt1*math.cos(phi1)
  p1y = pt1*math.sin(phi1)
  p1z = pt1*math.sinh(eta1)

  p2x = pt2*math.cos(phi2)
  p2y = pt2*math.sin(phi2)
  p2z = pt2*math.sinh(eta2)

  M = (E1+E2)**2 - (p1x+p2x)**2 - (p1y+p2y)**2 - (p1z+p2z)**2

  return str(math.sqrt(M))


f=open("dyevts5.txt", "r")
n=0
momenta=[]
evt_num=0
for line in f:
  n+=1
  toks = line.split("*")
#  if n > 20:
#    break

  if len(toks) < 2:
    continue

  try:
    if (not toks[1].strip().isdigit()):
      continue
    else:
      if evt_num != int(toks[1].strip()):
        momenta=[]
        evt_num = int(toks[1].strip())
    
      momenta += [toks[4], toks[5], toks[6], toks[7]]
    
      if len(momenta) > 5:
        print(",".join([M(momenta)]+momenta))
  except Exception as e:
    print(e)
    print(line)
    print(momenta)
    exit()
