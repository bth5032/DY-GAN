#pragma GCC diagnostic ignored "-Wsign-compare"

#include "TFile.h"
#include "TTree.h"
#include "TCut.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TH1.h"
#include "TChain.h"
#include "Math/VectorUtil.h"

#include "CMS3.h"

using namespace std;
using namespace tas;

int ScanChain(TChain *ch){

    int nEventsTotal = 0;
    int nEventsChain = ch->GetEntries();

    TFile *currentFile = 0;
    TObjArray *listOfFiles = ch->GetListOfFiles();
    TIter fileIter(listOfFiles);

    while ( (currentFile = (TFile*)fileIter.Next()) ) { 

        TFile *file = new TFile( currentFile->GetTitle() );
        TTree *tree = (TTree*)file->Get("Events");
        cms3.Init(tree);

        TString filename(currentFile->GetTitle());

        for( unsigned int event = 0; event < tree->GetEntriesFast(); ++event) {

            cms3.GetEntry(event);
            nEventsTotal++;

            // if (event > 100) break;

            CMS3::progress(nEventsTotal, nEventsChain);

            std::vector<int> zlep_indices;
            std::vector<int> other_indices;
            for (unsigned int igen = 0; igen < tas::genps_p4().size(); igen++){
                if (!tas::genps_isHardProcess()[igen]) continue;
                int id = tas::genps_id()[igen];
                int mid = tas::genps_id_mother()[igen];
                if (mid == 2212) continue; // We don't want stupid proton daughters
                if (abs(id) == 23) continue; // We don't care about the Z itself
                if (abs(mid) == 23) { // Count up z daughters
                    zlep_indices.push_back(igen);
                } else { // Count up other stuff (extra partons --> ngenjets?)
                    other_indices.push_back(igen);
                }
            }
            if (zlep_indices.size() == 2) {
                int idx1 = zlep_indices[0];
                int idx2 = zlep_indices[1];
                auto lep1 = tas::genps_p4()[idx1];
                auto lep2 = tas::genps_p4()[idx2];
                // for (auto idx : other_indices) {
                //     std::cout <<  " tas::genps_id()[idx]: " << tas::genps_id()[idx] <<  " tas::genps_id_mother()[idx]: " << tas::genps_id_mother()[idx] <<  std::endl;
                // }
                std::cout << (lep1+lep2).M() << ","
                    << other_indices.size() << ","
                    << lep1.E() << ","
                    << lep1.px() << ","
                    << lep1.py() << ","
                    << lep1.pz() << ","
                    << lep2.E() << ","
                    << lep2.px() << ","
                    << lep2.py() << ","
                    << lep2.pz()
                    << std::endl;
            }




        }//event loop

        delete file;
    }//file loop

    return 0;

}

