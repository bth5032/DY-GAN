## Instructions for jupyter notebooks on UAF

1. Set up a CMSSW environment: `cmsrel CMSSW_9_4_0_pre2; cd CMSSW_9_4_0_pre2; cmsenv; cd ..`
2. Start jupyter: `jupyter notebook --no-browser --port=8890` (or, to use a GPU on uaf-1, do
`singularity exec --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest jupyter notebook --no-browser --port=8890`)
3. Forward the port to your local computer:
```bash
# on local computer
# change port and hostname if necessary
ssh -N -f -L localhost:8890:localhost:8890 uaf-10.t2.ucsd.edu
```
4. Done.

