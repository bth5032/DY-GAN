import math
import numpy as np
import ROOT as r
import tqdm


"""
Generate 8-tuples of pair of E,pt,eta,phi (or E,px,py,pz) for Z decay products
"""

f1 = r.TF1("f1","sin(x)",0,3.14159);

# for i in tqdm.tqdm(range(1000000)):
for i in range(1000000):
    # Z boson
    eta = np.random.random()*4.8-2.4
    phi = np.random.random()*6.28-3.14
    pt = np.random.random()*30
    mz = np.random.normal(91,5)
    vz = r.TLorentzVector()
    vz.SetPtEtaPhiM(pt,eta,phi,mz)

    m_lep = 0.13
    e_lep =  0.5*mz
    p_lep = math.sqrt(e_lep*e_lep - m_lep*m_lep);
    phi_lep = np.random.random()*2*3.14159-3.14159
    theta_lep = f1.GetRandom()

    lep1 = r.TLorentzVector()
    lep2 = r.TLorentzVector()

    lep1.SetPxPyPzE( p_lep*math.cos(phi_lep)*math.sin(theta_lep), p_lep*math.sin(phi_lep)*math.sin(theta_lep), p_lep*math.cos(theta_lep), e_lep );
    lep2.SetPxPyPzE(-p_lep*math.cos(phi_lep)*math.sin(theta_lep),-p_lep*math.sin(phi_lep)*math.sin(theta_lep),-p_lep*math.cos(theta_lep), e_lep );

    lep1.Boost(vz.BoostVector());
    lep2.Boost(vz.BoostVector());

    # print "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(
    #         mz,
    #         lep1.E(), lep1.Pt(), lep1.Eta(), lep1.Phi(),
    #         lep2.E(), lep2.Pt(), lep2.Eta(), lep2.Phi(),
    #         )

    print "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(
            mz,
            lep1.E(), lep1.Px(), lep1.Py(), lep1.Pz(),
            lep2.E(), lep2.Px(), lep2.Py(), lep2.Pz(),
            )
