# GANs For Particle Physics Simulation
This project aims to train a GAN to produce particle physics events. Currently development is being undertaken in the ipython notebook `gan/notebooks/gan_skip_2.ipnb`

## DY-GAN
The goal of this repository is to train a GAN to produce Drell-Yan to dimuon events in conditions that replicate the proton-proton collisions at Large Hadron Collider and the CMS detector. The input dataset uses Monte Carlo integration and simulation of the CMS detector using the [pythia](http://home.thep.lu.se/~torbjorn/Pythia.html) event generator and the [Delphes](https://github.com/delphes/delphes) detector simulator. The dataset can be pulled from [here](http://uaf-10.t2.ucsd.edu/~bhashemi/GAN_Share/total_Zmumu_13TeV_PU20_v2.npa).

Once the dataset is loaded, you can launch your jupyter session and run the code in `gan_skip_2.ipynb`, a variety of configuration options are available which change the network architcture and training model. With the default configuration, you can expect the following performance level:

![alt text](http://uaf-10.t2.ucsd.edu/~bhashemi/GAN_Share/best_model.png)
