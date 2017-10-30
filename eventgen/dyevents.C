#include "TString.h"
#include "TChain.h"

void dyevents()
{
  int pdgId = 13; //Muons
  //int pdgId = 11; //Electrons

  TChain *ch = new TChain("t");
  ch->Add("/nfs-7/userdata/bhashemi/WWW_babies/WWW_v0.1.16/output/dy_m50_mgmlm_ext1*");
  ch->SetScanField(0);
  //pt,eta,phi TString query = "genPart_pdgId:genPart_p4.E():genPart_p4.pt():genPart_p4.eta():genPart_p4.phi()";
  TString query = "genPart_pdgId:genPart_p4.E():genPart_p4.X():genPart_p4.Y():genPart_p4.Z()";
  TString selection = Form("genPart_status==1 && Sum$(genPart_status==1 && abs(genPart_pdgId)==%i && abs(genPart_motherId) == 23 )==2 && (abs(genPart_pdgId)==%i && abs(genPart_motherId) == 23)", pdgId, pdgId);
  ch->GetEntries(selection);
  //ch->Scan(query , selection);
}
