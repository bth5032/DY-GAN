## Reading Delphes ntuples

Reference information for setting up Delphes with CMSSW is [here](https://twiki.cern.ch/twiki/bin/view/CMS/DelphesUPG).
Run through the setup section below once, then the conversion script can be used.

### Setup
```bash
cmsrel CMSSW_9_1_0_pre3
cd CMSSW_9_1_0_pre3
cmsenv
wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.4.1.tar.gz
curl -O -L https://github.com/delphes/delphes/archive/3.4.2pre03.tar.gz
tar -zxf 3.4.2pre03.tar.gz
cd delphes-3.4.2pre03
./configure
sed -i -e 's/c++0x/c++1y/g' Makefile
make -j 15
cd ../../
```

### Usage
* For the proper environment, please use the CMSSW release checked out by the setup previously.
* You may execute `python convert.py "/path/to/files/*.root"` to convert the ROOT files into a numpy array in `data.npa`.
* Modifying the script requires you to modify the `row` variable and also the `colnames` to match.
* Currently, you can do
```python
python convert.py "/home/users/namin/2017/gan/data/*.root"
```
