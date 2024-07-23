#define tree_reader_cxx
#include "tree_reader.h"
#include <TH1F.h>
#include <TObject.h>
#include <TKey.h>
#include "TLorentzVector.h"
#include <TFile.h>
#include <TROOT.h>
#include <iostream>
#include <fstream>
#include <string>
// We need to import such libraries to run the operations down below
void tree_reader::Loop()
{
    if (fChain == 0) return;
    Long64_t nbytes = 0, nb = 0;
    Long64_t nentries = fChain->GetEntriesFast();
    int DATA_MC_CHECK = 1; 
    /*
    a preliminary variable to see whether the file we are
    inspecting is data or MC file
    */
    TString check_name = (TString)m_destination; // to see the file name
    std::string fileName; // = "/home/uzobas/Desktop/2lep/zprime/ZAnalyses_HT.csv"
    if (check_name.Contains("ZPrime"))
    {
        fileName = "/home/uzobas/Desktop/2lep/zprime/ZPrime.csv";
        DATA_MC_CHECK = 1;
        /*
        If the file is a ZPrime or a any seeked file we determine the
        directory to set the csv file path
        */
        std::cout <<"SIGNAL"<<std::endl;
    }
    if (check_name.Contains("Zee_"))
    {
        fileName = "/home/uzobas/Desktop/2lep/zprime/background.csv";
        DATA_MC_CHECK=0;
        std::cout << "Background data detected" << std::endl;
        /*
        If the file is a Data or a any seeked file we determine the
        directory to set the csv file path
        */
    }
    std::ofstream outputFile;
    outputFile.open(fileName, std::ios::app);

    // Check if the file is open
    if (!outputFile.is_open())
    {
        std::cerr << "Error: Could not open file " << fileName << std::endl;
        return;
    }

    // Unrestricted

    // Histogram allocation
    TH1F *h_unr_el_n = new TH1F("n_el", "n_el", 10, 0, 10);
    TH1F *h_unr_el0_E = new TH1F("E_el0", "E_el0", 100, -10, 550);
    TH1F *h_unr_el1_E = new TH1F("E_el1", "E_el1", 100, -10, 550);
    TH1F *h_unr_el0_phi = new TH1F("phi_el0", "phi_el0", 100, -4, 4);
    TH1F *h_unr_el1_phi = new TH1F("phi_el1", "phi_el1", 100, -4, 4);
    TH1F *h_unr_el0_eta = new TH1F("eta_el0", "eta_el0", 100, -4, 4);
    TH1F *h_unr_el1_eta = new TH1F("eta_el1", "eta_el1", 100, -4, 4);
    TH1F *h_pt_el0 = new TH1F("pt_el0", "pt_el0", 100, 0, 200);
    TH1F *h_pt_el1 = new TH1F("pt_el1", "pt_el1", 100, 0, 200);
    TH1F *h_angle_e0e1 = new TH1F("angle_e0e1", "angle_e0e1", 100, 0, 3.14);

    TH1F *h_met_E = new TH1F("met_Et", "met_Et", 100, 0, 1000);
    TH1F *h_met_phi = new TH1F("met_phi", "met_phi", 100, -4, 4);

    for (Long64_t jentry = 0; jentry < nentries; jentry++)
    { //start of the entry loop, we see how many events there are and inspect each one them
        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        nb = fChain->GetEntry(jentry);
        nbytes += nb;

        // if (Cut(ientry) < 0) continue;

        float weight = 1.0;
        TLorentzVector el0;
        TLorentzVector el1;
        Double_t el0_E = 0, el1_E = 0, el0_phi = 0, el1_phi = 0, el0_eta = 0, el1_eta = 0, pt_el0 = 0, pt_el1 = 0;
        double delta_R = 0;
        double HT;
        h_met_E->Fill((met_et) / 1000.0, weight);
        h_met_phi->Fill(met_phi, weight);
        int electron_count = 0;
        /*
        Preliminary definitions of variables we will use during our analysis.
        It's important to set weight variable as a float 1.0 since we may be using
        this code to analyze Data and MC. By this way we can see if we have made an error.
        */
        
        for (int i = 0; i < lep_n; i++)
        {
            if ((*lep_type)[i] == 11)
            {
                if (electron_count == 0)
                {
                    electron_count++;
                    h_unr_el0_E->Fill(((*lep_E)[i]) / 1000.0);
                    el0_E = ((*lep_E)[i]) / 1000.0;
                    h_unr_el0_phi->Fill((*lep_phi)[i]);
                    el0_phi = (*lep_phi)[i];
                    h_unr_el0_eta->Fill((*lep_eta)[i]);
                    el0_eta = (*lep_eta)[i];
                    h_pt_el0->Fill((*lep_pt)[i] / 1000.0);
                    pt_el0 = (*lep_pt)[i] / 1000.0;
                }
                else if (electron_count == 1)
                {
                    electron_count++;
                    h_unr_el1_E->Fill(((*lep_E)[i]) / 1000.0);
                    el1_E = ((*lep_E)[i]) / 1000.0;
                    h_unr_el1_phi->Fill((*lep_phi)[i]);
                    el1_phi = (*lep_phi)[i];
                    h_unr_el1_eta->Fill((*lep_eta)[i]);
                    el1_eta = (*lep_eta)[i];
                    h_pt_el1->Fill((*lep_pt)[i] / 1000.0);
                    pt_el1 = (*lep_pt)[i] / 1000.0;
                }
                HT+=((*lep_pt)[i]);
            }
            if ((*lep_type)[i] == 13){
              HT+=((*lep_pt)[i]);
            }
        }
        /*
        We attain the values of vectors by (*vector)[iter]
        Also another important thing to note here is we use GeV for our anaylsis.
        Provided values from CERN are most of the time in MeV.
        To turn MeV to GeV simply multiply the MeV by 1e-3.
        */
        
        for(int i=0;i<jet_n;i++){
            HT+=((*jet_pt)[i]);

        }
        for(int i=0;i<photon_n;i++){
            HT+=((*photon_pt)[i]);

        }
        for(int i=0;i<tau_n;i++){
            HT+=((*tau_pt)[i]);

        }
        for(int i=0;i<largeRjet_n;i++){
            HT+=((*largeRjet_pt)[i]);

        }
        
        //Delta_R finding
        el0.SetPtEtaPhiE(pt_el0, el0_eta, el0_phi, el0_E);
        el1.SetPtEtaPhiE(pt_el1, el1_eta, el1_phi, el1_E);
        /*
        Setting the value of predefined TLorentzVectors el0 and el1
        */
        h_unr_el_n->Fill(electron_count);

        if (electron_count > 1) {
            delta_R = el0.DeltaR(el1);
            /*
            Using the already set values of el0 & el1 we determine the distance between them 
            */
        }
        
        
        
        
        
        outputFile << DATA_MC_CHECK << "," <<el0_E<<","<<el0_phi<<","<<el0_eta<<","<< pt_el0 << "," <<el1_E<<","<<el1_phi<<","<<el1_eta<<","<< pt_el1 << "," <<delta_R<<","<< jet_n << "," << met_et << ","<<HT<<std::endl;
        /*
        We write the values we had gathered during the iterations into the determined CSV file.
        */
    }

    // Close the file
    outputFile.close();
    TFile *of = TFile::Open(m_destination, "RECREATE");

    // Electron
    h_unr_el_n->Write();
    h_unr_el0_E->Write();
    h_unr_el1_E->Write();
    h_unr_el0_phi->Write();
    h_unr_el1_phi->Write();
    h_unr_el0_eta->Write();
    h_unr_el1_eta->Write();
    h_pt_el0->Write();
    h_pt_el1->Write();
    h_angle_e0e1->Write();
    h_met_E->Write();
    h_met_phi->Write();

    of->Close();
}
