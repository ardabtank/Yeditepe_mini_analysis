#define tree_reader_cxx
#include "tree_reader.h"
#include <TH1F.h>
#include <TKey.h>

void tree_reader::Loop()
{
  //   In a ROOT session, you can do:
  //      root> .L tree_reader.C
  //      root> tree_reader t
  //      root> t.GetEntry(12); // Fill t data members with entry number 12
  //      root> t.Show();       // Show values of entry 12
  //      root> t.Show(16);     // Read and show values of entry 16
  //      root> t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  // by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0)
    return;
  Long64_t nbytes = 0, nb = 0;
  Long64_t nentries = fChain->GetEntriesFast();
  // data to mc switch
  static int DATA_MC_CHECK=1;
  TString check_name= (TString) m_destination;
  
  if(check_name.Contains("data")==1){
  cout<<"Working with Data mode\n";
  DATA_MC_CHECK=1;
  }if(check_name.Contains("mc")==1){
  cout<<"Working with MC mode \n";
  DATA_MC_CHECK=0;
  }
  
  // unrestricted
  
  // electron
  TH1F *h_unr_el_n = new TH1F("n_el", "n_el", 10, 0, 10);
  TH1F *h_unr_el_E = new TH1F("E_el", "E_el", 100, -10, 550);
  TH1F *h_unr_el_phi = new TH1F("phi_el", "phi_el", 100, -4, 4);
  TH1F *h_unr_el_pt = new TH1F("pt_el", "pt_el", 100, 0, 200);
  TH1F *h_unr_el_eta = new TH1F("eta_el", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_unr_muon_n = new TH1F("n_mu", "n_mu", 10, 0, 10);
  TH1F *h_unr_muon_e = new TH1F("E_mu", "E_mu", 100, -10, 550);
  TH1F *h_unr_muon_phi = new TH1F("phi_mu", "phi_mu", 100, -4, 4);
  TH1F *h_unr_muon_pt = new TH1F("pt_mu", "pt_mu", 100, 0, 200);
  TH1F *h_unr_muon_eta = new TH1F("eta_mu", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_unr_jet_n = new TH1F("n_jet", "n_jet", 10, 0, 10); // degis
  TH1F *h_unr_jet_E = new TH1F("E_jet", "E_jet", 100, -10, 550);
  TH1F *h_unr_jet_phi = new TH1F("phi_jet", "phi_jet", 100, -4, 4);
  TH1F *h_unr_jet_pt = new TH1F("pt_jet", "pt_jet", 100, 0, 200);
  TH1F *h_unr_jet_eta = new TH1F("eta_jet", "eta_jet", 100, -4, 4);

  TH1F *h_unr_met_E = new TH1F("met_Et", "met_Et", 100, 0, 1000);
  TH1F *h_unr_met_phi = new TH1F("met_phi", "met_phi", 100, -4, 4);

  TH1F *h_unr_e_presence_met_E =
      new TH1F("met_Et_1el", "Presence of e^- MET Et", 100, 0, 1000);
  TH1F *h_unr_e_presence_met_phi =
      new TH1F("met_phi_1el", "Presence of e^- MET phi", 100, -4, 4);

  TH1F *h_unr_muon_presence_met_E =
      new TH1F("met_Et_1mu", "Presence of muon MET et(GeV)", 100, 0, 1000);
  TH1F *h_unr_muon_presence_met_phi =
      new TH1F("met_phi_1mu", "Presence of muon MET phi", 100, -4, 4);

  TH1F *h_unr_jet_presence_met_E =
      new TH1F("met_Et_1jet", "Presence of jet MET et(GeV)", 100, 0, 1000);
  TH1F *h_unr_jet_presence_met_phi =
      new TH1F("met_phi_1jet", "Presence of jet MET phi", 100, -4, 4);

  // for >= 2 electron

  TH1F *h_2el_el_n = new TH1F("n_el_2el", "n_el", 10, 0, 10);
  TH1F *h_2el_el_E = new TH1F("E_el_2el", "E_el", 100, -10, 550);
  TH1F *h_2el_el_phi = new TH1F("phi_el_2el", "phi_el", 100, -4, 4);
  TH1F *h_2el_el_pt = new TH1F("pt_el_2el", "pt_el", 100, 0, 200);
  TH1F *h_2el_el_eta = new TH1F("eta_el_2el", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_2el_muon_n = new TH1F("n_mu_2el", "n_mu", 10, 0, 10);
  TH1F *h_2el_muon_e = new TH1F("E_mu_2el", "E_mu", 100, -10, 550);
  TH1F *h_2el_muon_phi = new TH1F("phi_mu_2el", "phi_mu", 100, -4, 4);
  TH1F *h_2el_muon_pt = new TH1F("pt_mu_2el", "pt_mu", 100, 0, 200);
  TH1F *h_2el_muon_eta = new TH1F("eta_mu_2el", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_2el_jet_n = new TH1F("n_jet_2el", "n_jet", 10, 0, 10);
  TH1F *h_2el_jet_E = new TH1F("E_jet_2el", "E_jet", 100, -10, 550);
  TH1F *h_2el_jet_phi = new TH1F("phi_jet_2el", "phi_jet", 100, -4, 4);
  TH1F *h_2el_jet_pt = new TH1F("pt_jet_2el", "pt_jet", 100, 0, 200);
  TH1F *h_2el_jet_eta = new TH1F("eta_jet_2el", "eta_jet", 100, -4, 4);

  TH1F *h_2el_met_E = new TH1F("met_Et_2el", "met_Et", 100, 0, 1000);
  TH1F *h_2el_met_phi = new TH1F("met_phi_2el", "met_phi", 100, -4, 4);
  // since the MET et and phi will be the same for all measurements of >=2 el
  // only the met_Et and phi is enough

  // for >= 2 muon

  // muon
  TH1F *h_2muon_muon_n = new TH1F("n_mu_2mu", "n_mu", 10, 0, 10);
  TH1F *h_2muon_muon_e = new TH1F("E_mu _2mu", "E_mu", 100, -10, 550);
  TH1F *h_2muon_muon_phi = new TH1F("phi_mu_2mu", "phi_mu", 100, -4, 4);
  TH1F *h_2muon_muon_pt = new TH1F("pt_mu_2mu", "pt_mu", 100, 0, 200);
  TH1F *h_2muon_muon_eta = new TH1F("eta_mu_2mu", "eta_mu", 100, -4, 4);

  // electron
  TH1F *h_2muon_el_n = new TH1F("n_el_2mu", "n_el", 10, 0, 10);
  TH1F *h_2muon_el_E = new TH1F("E_el_2mu", "E_el", 100, -10, 550);
  TH1F *h_2muon_el_phi = new TH1F("phi_el_2mu", "phi_el", 100, -4, 4);
  TH1F *h_2muon_el_pt = new TH1F("pt_el_2mu", "pt_el", 100, 0, 200);
  TH1F *h_2muon_el_eta = new TH1F("eta_el_2mu", "eta_el", 100, -4, 4);

  // jet

  TH1F *h_2muon_jet_n = new TH1F("n_jet_2mu", "n_jet", 10, 0, 10);
  TH1F *h_2muon_jet_E = new TH1F("E_jet_2mu", "E_jet", 100, -10, 550);
  TH1F *h_2muon_jet_phi = new TH1F("phi_jet_2mu", "phi_jet", 100, -4, 4);
  TH1F *h_2muon_jet_pt = new TH1F("pt_jet_2mu", "pt_jet", 100, 0, 200);
  TH1F *h_2muon_jet_eta = new TH1F("eta_jet_2mu", "eta_jet", 100, -4, 4);

  // met

  TH1F *h_2muon_met_E = new TH1F("met_Et_2mu", "met_Et", 100, 0, 1000);
  TH1F *h_2muon_met_phi = new TH1F("met_phi_2mu", "met_phi", 100, -4, 4);

  // for >=1 jet
  TH1F *h_1jet_el_n = new TH1F("n_el_1j", "n_el", 10, 0, 10);
  TH1F *h_1jet_el_E = new TH1F("E_el_1j", "E_el", 100, -10, 550);
  TH1F *h_1jet_el_phi = new TH1F("phi_el_1j", "phi_el", 100, -4, 4);
  TH1F *h_1jet_el_pt = new TH1F("pt_el_1j", "pt_el", 100, 0, 200);
  TH1F *h_1jet_el_eta = new TH1F("eta_el_1j", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_1jet_muon_n = new TH1F("n_mu_1j", "n_mu", 10, 0, 10);
  TH1F *h_1jet_muon_e = new TH1F("E_mu_1j", "E_mu", 100, -10, 550);
  TH1F *h_1jet_muon_phi = new TH1F("phi_mu_1j", "phi_mu", 100, -4, 4);
  TH1F *h_1jet_muon_pt = new TH1F("pt_mu_1j", "pt_mu", 100, 0, 200);
  TH1F *h_1jet_muon_eta = new TH1F("eta_mu_1j", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_1jet_jet_n = new TH1F("n_jet_1j", "n_jet", 10, 0, 10);
  TH1F *h_1jet_jet_E = new TH1F("E_jet_1j", "E_jet", 100, -10, 550);
  TH1F *h_1jet_jet_phi = new TH1F("phi_jet_1j", "phi_jet", 100, -4, 4);
  TH1F *h_1jet_jet_pt = new TH1F("pt_jet_1j", "pt_jet", 100, 0, 200);
  TH1F *h_1jet_jet_eta = new TH1F("eta_jet_1j", "eta_jet", 100, -4, 4);

  TH1F *h_1jet_met_E = new TH1F("met_Et_1j", "met_Et", 100, 0, 1000);
  TH1F *h_1jet_met_phi = new TH1F("met_phi_1j", "met_phi", 100, -4, 4);

  // for >=2 jet
  TH1F *h_2jet_el_n = new TH1F("n_el_2j", "n_el", 10, 0, 10);
  TH1F *h_2jet_el_E = new TH1F("E_el_2j", "E_el", 100, -10, 550);
  TH1F *h_2jet_el_phi = new TH1F("phi_el_2j", "phi_el", 100, -4, 4);
  TH1F *h_2jet_el_pt = new TH1F("pt_el_2j", "pt_el", 100, 0, 200);
  TH1F *h_2jet_el_eta = new TH1F("eta_el_2j", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_2jet_muon_n = new TH1F("n_mu_2j", "n_mu", 10, 0, 10);
  TH1F *h_2jet_muon_e = new TH1F("E_mu_2j", "E_mu", 100, -10, 550);
  TH1F *h_2jet_muon_phi = new TH1F("phi_mu_2j", "phi_mu", 100, -4, 4);
  TH1F *h_2jet_muon_pt = new TH1F("pt_mu_2j", "pt_mu", 100, 0, 200);
  TH1F *h_2jet_muon_eta = new TH1F("eta_mu_2j", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_2jet_jet_n = new TH1F("n_jet_2j", "n_jet", 10, 0, 10);
  TH1F *h_2jet_jet_E = new TH1F("E_jet_2j", "E_jet", 100, -10, 550);
  TH1F *h_2jet_jet_phi = new TH1F("phi_jet_2j", "phi_jet", 100, -4, 4);
  TH1F *h_2jet_jet_pt = new TH1F("pt_jet_2j", "pt_jet", 100, 0, 200);
  TH1F *h_2jet_jet_eta = new TH1F("eta_jet_2j", "eta_jet", 100, -4, 4);

  TH1F *h_2jet_met_E = new TH1F("met_Et_2j", "met_Et", 100, 0, 1000);
  TH1F *h_2jet_met_phi = new TH1F("met_phi_2j", "met_phi", 100, -4, 4);
  // unr means unrestricted,
  
  TH1F* h_weight = new TH1F("weight","weight",150,-10,10);
  
  //0 lep
  
  TH1F *h_no_lep_el_n = new TH1F("n_el_0lep", "n_el", 10, 0, 10);
  TH1F *h_no_lep_el_E = new TH1F("E_el_0lep", "E_el", 100, -10, 550);
  TH1F *h_no_lep_el_phi = new TH1F("phi_el_0lep", "phi_el", 100, -4, 4);
  TH1F *h_no_lep_el_pt = new TH1F("pt_el_0lep", "pt_el", 100, 0, 200);
  TH1F *h_no_lep_el_eta = new TH1F("eta_el_0lep", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_no_lep_muon_n = new TH1F("n_mu_0lep", "n_mu", 10, 0, 10);
  TH1F *h_no_lep_muon_e = new TH1F("E_mu_0lep", "E_mu", 100, -10, 550);
  TH1F *h_no_lep_muon_phi = new TH1F("phi_mu_0lep", "phi_mu", 100, -4, 4);
  TH1F *h_no_lep_muon_pt = new TH1F("pt_mu_0lep", "pt_mu", 100, 0, 200);
  TH1F *h_no_lep_muon_eta = new TH1F("eta_mu_0lep", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_no_lep_jet_n = new TH1F("n_jet_0lep", "n_jet", 10, 0, 10);
  TH1F *h_no_lep_jet_E = new TH1F("E_jet_0lep", "E_jet", 100, -10, 550);
  TH1F *h_no_lep_jet_phi = new TH1F("phi_jet_0lep", "phi_jet", 100, -4, 4);
  TH1F *h_no_lep_jet_pt = new TH1F("pt_jet_0lep", "pt_jet", 100, 0, 200);
  TH1F *h_no_lep_jet_eta = new TH1F("eta_jet_0lep", "eta_jet", 100, -4, 4);

  TH1F *h_no_lep_met_E = new TH1F("met_Et_0lep", "met_Et", 100, 0, 1000);
  TH1F *h_no_lep_met_phi = new TH1F("met_phi_0lep", "met_phi", 100, -4, 4);
  
  //0 el
  
  TH1F *h_no_el_el_n = new TH1F("n_el_0el", "n_el", 10, 0, 10);
  TH1F *h_no_el_el_E = new TH1F("E_el_0el", "E_el", 100, -10, 550);
  TH1F *h_no_el_el_phi = new TH1F("phi_el_0el", "phi_el", 100, -4, 4);
  TH1F *h_no_el_el_pt = new TH1F("pt_el_0el", "pt_el", 100, 0, 200);
  TH1F *h_no_el_el_eta = new TH1F("eta_el_0el", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_no_el_muon_n = new TH1F("n_mu_0el", "n_mu", 10, 0, 10);
  TH1F *h_no_el_muon_e = new TH1F("E_mu_0el", "E_mu", 100, -10, 550);
  TH1F *h_no_el_muon_phi = new TH1F("phi_mu_0el", "phi_mu", 100, -4, 4);
  TH1F *h_no_el_muon_pt = new TH1F("pt_mu_0el", "pt_mu", 100, 0, 200);
  TH1F *h_no_el_muon_eta = new TH1F("eta_mu_0el", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_no_el_jet_n = new TH1F("n_jet_0el", "n_jet", 10, 0, 10);
  TH1F *h_no_el_jet_E = new TH1F("E_jet_0el", "E_jet", 100, -10, 550);
  TH1F *h_no_el_jet_phi = new TH1F("phi_jet_0el", "phi_jet", 100, -4, 4);
  TH1F *h_no_el_jet_pt = new TH1F("pt_jet_0el", "pt_jet", 100, 0, 200);
  TH1F *h_no_el_jet_eta = new TH1F("eta_jet_0el", "eta_jet", 100, -4, 4);

  TH1F *h_no_el_met_E = new TH1F("met_Et_0el", "met_Et", 100, 0, 1000);
  TH1F *h_no_el_met_phi = new TH1F("met_phi_0el", "met_phi", 100, -4, 4);
  
  //0 muon
  
  TH1F *h_no_mu_el_n = new TH1F("n_el_0mu", "n_el", 10, 0, 10);
  TH1F *h_no_mu_el_E = new TH1F("E_el_0mu", "E_el", 100, -10, 550);
  TH1F *h_no_mu_el_phi = new TH1F("phi_el_0mu", "phi_el", 100, -4, 4);
  TH1F *h_no_mu_el_pt = new TH1F("pt_el_0mu", "pt_el", 100, 0, 200);
  TH1F *h_no_mu_el_eta = new TH1F("eta_el_0mu", "eta_el", 100, -4, 4);
  // muon
  TH1F *h_no_mu_muon_n = new TH1F("n_mu_0mu", "n_mu", 10, 0, 10);
  TH1F *h_no_mu_muon_e = new TH1F("E_mu_0mu", "E_mu", 100, -10, 550);
  TH1F *h_no_mu_muon_phi = new TH1F("phi_mu_0mu", "phi_mu", 100, -4, 4);
  TH1F *h_no_mu_muon_pt = new TH1F("pt_mu_0mu", "pt_mu", 100, 0, 200);
  TH1F *h_no_mu_muon_eta = new TH1F("eta_mu_0mu", "eta_mu", 100, -4, 4);

  // jet
  TH1F *h_no_mu_jet_n = new TH1F("n_jet_0mu", "n_jet", 10, 0, 10);
  TH1F *h_no_mu_jet_E = new TH1F("E_jet_0mu", "E_jet", 100, -10, 550);
  TH1F *h_no_mu_jet_phi = new TH1F("phi_jet_0mu", "phi_jet", 100, -4, 4);
  TH1F *h_no_mu_jet_pt = new TH1F("pt_jet_0mu", "pt_jet", 100, 0, 200);
  TH1F *h_no_mu_jet_eta = new TH1F("eta_jet_0mu", "eta_jet", 100, -4, 4);

  TH1F *h_no_mu_met_E = new TH1F("met_Et_0mu", "met_Et", 100, 0, 1000);
  TH1F *h_no_mu_met_phi = new TH1F("met_phi_0mu", "met_phi", 100, -4, 4);
  

  // how to save a histogram to a root file ?
  // short deger = met_phi; //degis
  //  double dut = (double)((int)met_phi*100)/100;

  // Aşağıdaki  event loop ... : dosyadaki bütün  entryler üzerinden dönüyor
  for (Long64_t jentry = 0; jentry < nentries; jentry++)
  {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    // lep_e->at(0) lep_n bunu kullan
    // (*lep_e)[int x]
    
    // Bizim işimiz burada başlıyo....
    // Elektron histogramlarını doldur....
    static float weight=1.0;
    if(DATA_MC_CHECK ==1){
    float weight=1.0;
    }
    if(DATA_MC_CHECK==0){
    float scalingfactor = scaleFactor_PILEUP*scaleFactor_ELE*scaleFactor_MUON*scaleFactor_PHOTON*scaleFactor_TAU*scaleFactor_BTAG*scaleFactor_LepTRIGGER*scaleFactor_PhotonTRIGGER;
    weight = ((float) (XSection*(1e+3)*scalingfactor*10*mcWeight)/((float)(SumWeights)));// WEIGHT FOR MC
    
    }

  
    
    h_weight->Fill(weight);
    
    //float weight= 1.0; // WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA WEIGHT FOR DATA
    h_unr_met_E->Fill((met_et) / float(1000),weight);
    h_unr_met_phi->Fill(met_phi,weight);
    
    // Serhat: Bütün işi üstteki gibi kocaman bir if blok içine almaktansa :
    // Burada şunu desek ? : if(lep_n == 0) continue;
    // yani zaten içinde lepton olmayan bir eventi dogrudan çöpe atacağız..
    //  bunları ta en başta çöpe atıp kurtulsak ? sonra kalanlarla devam etsek ?
    int electron_count = 0;
    int muon_count = 0;
    for (int counter = 0; counter < lep_n; counter++)
    {
      if ((*lep_type)[counter] == 11)
      {
        float pt_el= (*lep_pt)[counter] / 1000.0;
        float eta_el=(*lep_eta)[counter];
        bool trig = ( (trigE)==1 && (trigM)==0) || ( (trigE)==0 && (trigM)==1  );
        if( (pt_el)>7.0 && fabs(eta_el)<2.47 && !(fabs(eta_el)<1.52 && fabs(eta_el)>1.37) && trig ){
        electron_count += 1;
        
        h_unr_el_eta->Fill((*lep_eta)[counter],weight);
        h_unr_el_phi->Fill((*lep_phi)[counter],weight);
        h_unr_el_pt->Fill((*lep_pt)[counter] / 1000.0,weight);
        h_unr_el_E->Fill((*lep_E)[counter] / 1000.0,weight);
        }
      }

      if ((*lep_type)[counter] == 13)
      {
        if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.5){
        muon_count += 1;

        h_unr_muon_eta->Fill((*lep_eta)[counter],weight);
        h_unr_muon_phi->Fill((*lep_phi)[counter],weight);
        h_unr_muon_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
        h_unr_muon_e->Fill((float((*lep_E)[counter]) / 1000.0),weight);
        }
      }
    }

    // fill regardless for values

    // Kalan işler :

    // All values for el_n >=2

    // All Values for muon_n >=2

    // All values for jet_n >=1

    // All values for jet_n >=2

    // jet
    int jet_count = 0;
    for (int count = 0; count < jet_n; count++)
    {
      if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
      jet_count++;
      h_unr_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
      h_unr_jet_eta->Fill(((*jet_eta)[count]),weight);
      h_unr_jet_phi->Fill(((*jet_phi)[count]),weight);
      h_unr_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
      }
    }
    // Fill counts without discrimination
    h_unr_el_n->Fill(electron_count,weight);
    h_unr_muon_n->Fill(muon_count,weight);
    h_unr_jet_n->Fill(jet_n,weight);
    // Fill met if n_el is greater than 0
    if (electron_count != 0)
    {
      h_unr_e_presence_met_E->Fill(((met_et) / float(1000)),weight);
      h_unr_e_presence_met_phi->Fill(met_phi,weight);
    }
    // Fill met if n_mu is greater than 0
    if (muon_count != 0)
    {
      h_unr_muon_presence_met_E->Fill(((met_et) / float(1000)),weight);
      h_unr_muon_presence_met_phi->Fill(met_phi,weight);
    }
    // Fill met if n_jet is greater than 0
    if (jet_count != 0)
    {
      h_unr_jet_presence_met_E->Fill(((met_et) / float(1000)),weight);
      h_unr_jet_presence_met_phi->Fill(met_phi,weight);
    }

    // for >=2 electron

    if (electron_count >= 2)
    {
      h_2el_met_E->Fill(((met_et) / float(1000)),weight);
      h_2el_met_phi->Fill(met_phi,weight);

      for (int counter = 0; counter < lep_n; counter++)
      {
        if ((*lep_type)[counter] == 11)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.47){

            h_2el_el_eta->Fill((*lep_eta)[counter],weight);
            h_2el_el_phi->Fill((*lep_phi)[counter],weight);
            h_2el_el_pt->Fill(((*lep_pt)[counter] / 1000.0),weight);
            h_2el_el_E->Fill(((*lep_E)[counter] / 1000.0),weight);
          }
        }

        if ((*lep_type)[counter] == 13)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.5){
            h_2el_muon_eta->Fill(((*lep_eta)[counter]),weight);
            h_2el_muon_phi->Fill(((*lep_phi)[counter]),weight);
            h_2el_muon_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
            h_2el_muon_e->Fill((float((*lep_E)[counter]) / 1000.0),weight);
          }
        }
      }

      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
          h_2el_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
          h_2el_jet_eta->Fill(((*jet_eta)[count]),weight);
          h_2el_jet_phi->Fill(((*jet_phi)[count]),weight);
          h_2el_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_2el_el_n->Fill(electron_count,weight);
      h_2el_muon_n->Fill(muon_count,weight);
      h_2el_jet_n->Fill(jet_n,weight);
    }
    // for >= 2 muon
    if (muon_count >= 2)
    {
      h_2muon_met_E->Fill(((met_et) / float(1000)),weight);
      h_2muon_met_phi->Fill(met_phi,weight);

      for (int counter = 0; counter < lep_n; counter++)
      {
        if ((*lep_type)[counter] == 11)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.47){
          h_2muon_el_eta->Fill(((*lep_eta)[counter]),weight);
          h_2muon_el_phi->Fill(((*lep_phi)[counter]),weight);
          h_2muon_el_pt->Fill(((*lep_pt)[counter] / 1000.0),weight);
          h_2muon_el_E->Fill(((*lep_E)[counter] / 1000.0),weight);
          }
        }

        if ((*lep_type)[counter] == 13)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.5){
          h_2muon_muon_eta->Fill(((*lep_eta)[counter]),weight);
          h_2muon_muon_phi->Fill(((*lep_phi)[counter]),weight);
          h_2muon_muon_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
          h_2muon_muon_e->Fill((float((*lep_E)[counter]) / 1000.0),weight);
          }
      }
      }
      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
        h_2muon_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
        h_2muon_jet_eta->Fill(((*jet_eta)[count]),weight);
        h_2muon_jet_phi->Fill(((*jet_phi)[count]),weight);
        h_2muon_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_2muon_el_n->Fill(electron_count,weight);
      h_2muon_muon_n->Fill(muon_count,weight);
      h_2muon_jet_n->Fill(jet_n,weight);
    }

    // for jet >=1
    if (jet_count >= 1)
    {
      h_1jet_met_E->Fill(((met_et) / float(1000)),weight);
      h_1jet_met_phi->Fill(met_phi,weight);

      for (int counter = 0; counter < lep_n; counter++)
      {
        if ((*lep_type)[counter] == 11)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.47){
          h_1jet_el_eta->Fill(((*lep_eta)[counter]),weight);
          h_1jet_el_phi->Fill(((*lep_phi)[counter]),weight);
          h_1jet_el_pt->Fill(((*lep_pt)[counter] / 1000.0),weight);
          h_1jet_el_E->Fill(((*lep_E)[counter] / 1000.0),weight);
          }
        }

        if ((*lep_type)[counter] == 13)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.5){
          h_1jet_muon_eta->Fill(((*lep_eta)[counter]),weight);
          h_1jet_muon_phi->Fill(((*lep_phi)[counter]),weight);
          h_1jet_muon_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
          h_1jet_muon_e->Fill((float((*lep_E)[counter]) / 1000.0),weight);
          }
        }
      }

      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
        h_1jet_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
        h_1jet_jet_eta->Fill(((*jet_eta)[count]),weight);
        h_1jet_jet_phi->Fill(((*jet_phi)[count]),weight);
        h_1jet_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_1jet_el_n->Fill(electron_count,weight);
      h_1jet_muon_n->Fill(muon_count,weight);
      h_1jet_jet_n->Fill(jet_n,weight);
    }
    // for jet >=2
    if (jet_count >= 2)
    {
      h_2jet_met_E->Fill(((met_et) / float(1000)),weight);
      h_2jet_met_phi->Fill(met_phi,weight);

      for (int counter = 0; counter < lep_n; counter++)
      {
        if ((*lep_type)[counter] == 11)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.47){
          h_2jet_el_eta->Fill(((*lep_eta)[counter]),weight);
          h_2jet_el_phi->Fill(((*lep_phi)[counter]),weight);
          h_2jet_el_pt->Fill(((*lep_pt)[counter] / 1000.0),weight);
          h_2jet_el_E->Fill(((*lep_E)[counter] / 1000.0),weight);
          }
        }

        if ((*lep_type)[counter] == 13)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.5){
          h_2jet_muon_eta->Fill(((*lep_eta)[counter]),weight);
          h_2jet_muon_phi->Fill(((*lep_phi)[counter]),weight);
          h_2jet_muon_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
          h_2jet_muon_e->Fill((float((*lep_E)[counter]) / 1000.0),weight);
          }
        }
      }

      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
        h_2jet_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
        h_2jet_jet_eta->Fill(((*jet_eta)[count]),weight);
        h_2jet_jet_phi->Fill(((*jet_phi)[count]),weight);
        h_2jet_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_2jet_el_n->Fill(electron_count,weight);
      h_2jet_muon_n->Fill(muon_count,weight);
      h_2jet_jet_n->Fill(jet_n,weight);
    }
    //for 0 lep
    if (lep_n == 0)
    {
      h_no_lep_met_E->Fill(((met_et) / float(1000)),weight);
      h_no_lep_met_phi->Fill(met_phi,weight);

          h_no_lep_el_eta->Fill(0);
          h_no_lep_el_phi->Fill(0);
          h_no_lep_el_pt->Fill(0);
          h_no_lep_el_E->Fill(0);
       
          h_no_lep_muon_eta->Fill(0);
          h_no_lep_muon_phi->Fill(0);
          h_no_lep_muon_pt->Fill(0);
          h_no_lep_muon_e->Fill(0);
       

      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
        h_no_lep_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
        h_no_lep_jet_eta->Fill(((*jet_eta)[count]),weight);
        h_no_lep_jet_phi->Fill(((*jet_phi)[count]),weight);
        h_no_lep_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_no_lep_el_n->Fill(0);
      h_no_lep_muon_n->Fill(0);
      h_no_lep_jet_n->Fill(jet_n);
    }
    // for 0 el
    if (electron_count == 0)
    {
      h_no_el_met_E->Fill(((met_et) / float(1000)),weight);
      h_no_el_met_phi->Fill(met_phi,weight);
        for (int counter = 0; counter < lep_n; counter++)
      {
          h_no_el_el_eta->Fill(0);
          h_no_el_el_phi->Fill(0);
          h_no_el_el_pt->Fill(0);
          h_no_el_el_E->Fill(0);
        if ((*lep_type)[counter] == 13)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.5){
          h_no_el_muon_eta->Fill(((*lep_eta)[counter]),weight);
          h_no_el_muon_phi->Fill(((*lep_phi)[counter]),weight);
          h_no_el_muon_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
          h_no_el_muon_e->Fill((float((*lep_E)[counter]) / 1000.0),weight);
          }
       
        }
      }
    
      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
        h_no_el_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
        h_no_el_jet_eta->Fill(((*jet_eta)[count]),weight);
        h_no_el_jet_phi->Fill(((*jet_phi)[count]),weight);
        h_no_el_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_no_el_el_n->Fill(0);
      h_no_el_muon_n->Fill(muon_count);
      h_no_el_jet_n->Fill(jet_n);
    }
  // for 0 mu
    if (muon_count == 0)
    {
      h_no_mu_met_E->Fill(((met_et) / float(1000)),weight);
      h_no_mu_met_phi->Fill(met_phi,weight);
        for (int counter = 0; counter < lep_n; counter++)
      {
        if ((*lep_type)[counter] == 11)
        {
          if( ((*lep_pt)[counter] / 1000.0)>7.0 && (*lep_eta)[counter]<2.47){
          h_no_mu_el_eta->Fill(((*lep_eta)[counter]),weight);
          h_no_mu_el_phi->Fill(((*lep_phi)[counter]),weight);
          h_no_mu_el_pt->Fill(((float((*lep_pt)[counter])) / 1000.0),weight);
          h_no_mu_el_E->Fill((float((*lep_E)[counter]) / 1000.0),weight);
          }
        }
          h_no_mu_muon_eta->Fill(0);
          h_no_mu_muon_phi->Fill(0);
          h_no_mu_muon_pt->Fill(0);
          h_no_mu_muon_e->Fill(0);
       

      for (int count = 0; count < jet_n; count++)
      {
        if((((float((*jet_pt)[count])) / 1000)>25.0 && ((*jet_jvt)[count])>0.59)){
        h_no_mu_jet_pt->Fill(((float((*jet_pt)[count])) / 1000),weight);
        h_no_mu_jet_eta->Fill(((*jet_eta)[count]),weight);
        h_no_mu_jet_phi->Fill(((*jet_phi)[count]),weight);
        h_no_mu_jet_E->Fill((float((*jet_E)[count]) / 1000),weight);
        }
      }

      h_no_mu_el_n->Fill(electron_count);
      h_no_mu_muon_n->Fill(0);
      h_no_mu_jet_n->Fill(jet_n);
    }
  }
}
  // Bütün histogramları output dosyasına yaz ...

  TFile *of = TFile::Open(m_destination, "RECREATE");

  // Electron
  // e^-
  h_unr_el_n->Write();
  h_unr_el_E->Write();
  h_unr_el_phi->Write();
  h_unr_el_pt->Write();
  h_unr_el_eta->Write();
  // muon
  h_unr_muon_n->Write();
  h_unr_muon_e->Write();
  h_unr_muon_phi->Write();
  h_unr_muon_pt->Write();
  h_unr_muon_eta->Write();

  // jet
  h_unr_jet_n->Write();
  h_unr_jet_E->Write();
  h_unr_jet_phi->Write();
  h_unr_jet_pt->Write();
  h_unr_jet_eta->Write();

  // met
  h_unr_met_E->Write();
  h_unr_met_phi->Write();
  h_unr_e_presence_met_E->Write();
  h_unr_e_presence_met_phi->Write();
  h_unr_muon_presence_met_E->Write();
  h_unr_muon_presence_met_phi->Write();
  h_unr_jet_presence_met_E->Write();
  h_unr_jet_presence_met_phi->Write();
  
  // for >=2 electrons
  // e^-
  h_2el_el_n->Write();
  h_2el_el_E->Write();
  h_2el_el_phi->Write();
  h_2el_el_pt->Write();
  h_2el_el_eta->Write();
  // muon
  h_2el_muon_n->Write();
  h_2el_muon_e->Write();
  h_2el_muon_phi->Write();
  h_2el_muon_pt->Write();
  h_2el_muon_eta->Write();

  // jet
  h_2el_jet_n->Write();
  h_2el_jet_E->Write();
  h_2el_jet_phi->Write();
  h_2el_jet_pt->Write();
  h_2el_jet_eta->Write();

  // met
  h_2el_met_E->Write();
  h_2el_met_phi->Write();

  // for >=2 muons

  // muon
  h_2muon_muon_n->Write();
  h_2muon_muon_e->Write();
  h_2muon_muon_phi->Write();
  h_2muon_muon_pt->Write();
  h_2muon_muon_eta->Write();

  // e^-

  h_2muon_el_n->Write();
  h_2muon_el_E->Write();
  h_2muon_el_phi->Write();
  h_2muon_el_pt->Write();
  h_2muon_el_eta->Write();

  // jet

  h_2muon_jet_n->Write();
  h_2muon_jet_E->Write();
  h_2muon_jet_phi->Write();
  h_2muon_jet_pt->Write();
  h_2muon_jet_eta->Write();

  // met

  h_2muon_met_E->Write();
  h_2muon_met_phi->Write();

  // for >=1 jet
  // e^-
  h_1jet_el_n->Write();
  h_1jet_el_E->Write();
  h_1jet_el_phi->Write();
  h_1jet_el_pt->Write();
  h_1jet_el_eta->Write();
  // muon
  h_1jet_muon_n->Write();
  h_1jet_muon_e->Write();
  h_1jet_muon_phi->Write();
  h_1jet_muon_pt->Write();
  h_1jet_muon_eta->Write();

  // jet
  h_1jet_jet_n->Write();
  h_1jet_jet_E->Write();
  h_1jet_jet_phi->Write();
  h_1jet_jet_pt->Write();
  h_1jet_jet_eta->Write();

  // met
  h_1jet_met_E->Write();
  h_1jet_met_phi->Write();

  // for >=2 jet
  // e^-
  h_2jet_el_n->Write();
  h_2jet_el_E->Write();
  h_2jet_el_phi->Write();
  h_2jet_el_pt->Write();
  h_2jet_el_eta->Write();
  // muon
  h_2jet_muon_n->Write();
  h_2jet_muon_e->Write();
  h_2jet_muon_phi->Write();
  h_2jet_muon_pt->Write();
  h_2jet_muon_eta->Write();

  // jet
  h_2jet_jet_n->Write();
  h_2jet_jet_E->Write();
  h_2jet_jet_phi->Write();
  h_2jet_jet_pt->Write();
  h_2jet_jet_eta->Write();

  // met
  h_2jet_met_E->Write();
  h_2jet_met_phi->Write();
  
  
  //0 lep
  h_no_lep_met_E->Write();
  h_no_lep_el_E->Write();
  h_no_lep_el_phi->Write();
  h_no_lep_el_pt->Write();
  h_no_lep_el_eta->Write();
  // muon
  h_no_lep_muon_n->Write();
  h_no_lep_muon_e->Write();
  h_no_lep_muon_phi->Write();
  h_no_lep_muon_pt->Write();
  h_no_lep_muon_eta->Write();

  // jet
  h_no_lep_jet_n->Write();
  h_no_lep_jet_E->Write();
  h_no_lep_jet_phi->Write();
  h_no_lep_jet_pt->Write();
  h_no_lep_jet_eta->Write();

  // met
  h_no_lep_met_E->Write();
  h_no_lep_met_phi->Write();
  
  //0 el
  
  h_no_el_el_n->Write();
  h_no_el_el_E->Write();
  h_no_el_el_phi->Write();
  h_no_el_el_pt->Write();
  h_no_el_el_eta->Write();
  // muon
  h_no_el_muon_n->Write();
  h_no_el_muon_e->Write();
  h_no_el_muon_phi->Write();
  h_no_el_muon_pt->Write();
  h_no_el_muon_eta->Write();

  // jet
  h_no_el_jet_n->Write();
  h_no_el_jet_E->Write();
  h_no_el_jet_phi->Write();
  h_no_el_jet_pt->Write();
  h_no_el_jet_eta->Write();

  // met
  h_no_el_met_E->Write();
  h_no_el_met_phi->Write();
  
  //0 muon
  
  h_no_mu_el_n->Write();
  h_no_mu_el_E->Write();
  h_no_mu_el_phi->Write();
  h_no_mu_el_pt->Write();
  h_no_mu_el_eta->Write();
  // muon
  h_no_mu_muon_n->Write();
  h_no_mu_muon_e->Write();
  h_no_mu_muon_phi->Write();
  h_no_mu_muon_pt->Write();
  h_no_mu_muon_eta->Write();

  // jet
  h_no_mu_jet_n->Write();
  h_no_mu_jet_E->Write();
  h_no_mu_jet_phi->Write();
  h_no_mu_jet_pt->Write();
  h_no_mu_jet_eta->Write();

  // met
  h_no_mu_met_E->Write();
  h_no_mu_met_phi->Write();
  
  h_weight->Write();

  of->Close();
}
