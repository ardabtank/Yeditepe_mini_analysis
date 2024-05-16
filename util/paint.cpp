#include <iostream>
#include <string> // Include for std::string
#include <TFile.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TKey.h> // Include for TKey
#include <TSystem.h> // Include for gSystem

void bugucu(const std::string& file1, const std::string& file2) {
    TCanvas *canvas = new TCanvas("canvas", "Matching Histograms", 800, 600);

    // Convert std::string to const char*
    const char* file1_cstr = file1.c_str();
    const char* file2_cstr = file2.c_str();

    TFile *inputFile1 = TFile::Open(file1_cstr);
    TFile *inputFile2 = TFile::Open(file2_cstr);

    if (!inputFile1 || !inputFile2) {
        std::cerr << "Error: Unable to open input files!" << std::endl;
        return;
    }

    TFile *outputFile = TFile::Open("histo_on_histo.root", "RECREATE");
    if (!outputFile) {
        std::cerr << "Error: Unable to create output file!" << std::endl;
        return;
    }

    TString outputDirectory = "./Drawn_Histograms/";
    gSystem->mkdir(outputDirectory, true); // Create the directory if it doesn't exist

    TIter next1(inputFile1->GetListOfKeys());
    TKey *key1;
    int i=0;
    while ((key1 = (TKey*)next1())) {
        if (key1->ReadObj()->InheritsFrom("TH1")) {
            i++;
            TH1 *hist1 = (TH1*)key1->ReadObj();
            TH1 *hist2 = (TH1*)inputFile2->Get(hist1->GetName());
            if (hist2) {
            long hist1entries= hist1->GetMaximum();
            long hist2entries= hist2->GetMaximum();
            if(hist1entries>hist2entries){
                hist2->SetLineColor(kRed);
                hist1->Draw();
                hist2->Draw("SAME");
                canvas->Update();
                TString final_name = Form("%s/%d_%s.png", outputDirectory.Data(), i, hist1->GetName());
                canvas->SaveAs(final_name);
            }
            if(hist2entries>hist1entries){
                hist2->SetLineColor(kRed);
                hist2->Draw();
                hist1->Draw("SAME");
                canvas->Update();
                TString final_name = Form("%s/%d_%s.png", outputDirectory.Data(), i, hist1->GetName());
                canvas->SaveAs(final_name);
            }
            //red one is always the MonteCarlo
        }
    }
    }

    // Save canvas and close files
    outputFile->Close();
    inputFile1->Close();
    inputFile2->Close();
}

int main(int argc, char* argv[]) {
    if (argc != 3 ) {
        std::cerr << "Usage: " << argv[0] << " <file1> <file2>\n";
        return 1;
    }
    const std::string file1 = argv[1]; // Use std::string
    const std::string file2 = argv[2]; // Use std::string

    bugucu(file1, file2);

    return 0;
}

