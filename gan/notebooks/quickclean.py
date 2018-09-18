f = open("tmp_run2.txt", "r")

run=4
epoch=500

for line in f:
  line=line[:-1]
  if "Getting" in line:
    pass
  elif "===" in line:
    run=line[:line.index('=')]
    epoch=500
  else:
    cleanline = line[:line.index(',')]+" "+line[line.index("=")+1:line.index(")")]
    print "%s %s %d" % (cleanline, run, epoch)
    epoch+=500

