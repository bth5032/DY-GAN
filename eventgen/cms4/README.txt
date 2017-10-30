### Overview

Prints out gen information from CMS4 files, specified in `doAll.C` in the TChain,
with selection logic in `ScanChain.C`.

### Instructions

```bash
root -b -q -l doAll.C > out.csv
sed -i '1,2d' out.csv # delete stupid lines from ROOT
```

