{

    gROOT->ProcessLine(".L CMS3.cc+");
    gROOT->ProcessLine(".L ScanChain.C+");

    TChain *ch = new TChain("Events");
    ch->Add("/hadoop/cms/store/user/namin/ProjectMetis/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer17MiniAOD-92X_upgrade2017_realistic_v10_ext1-v1_MINIAODSIM_CMS4_V00-00-06/merged_ntuple_1.root");

    ScanChain(ch);

}

