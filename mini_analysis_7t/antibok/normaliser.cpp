#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include <sstream>

// Public variable to store the cross-section value
double CrossSection = 0.0;

// Function to perform normalization run on histograms
void performNormalizationRun(const std::string& inputFileName, double luminosity, const std::string& outputDir) {
    // Open the input file
    std::cout << "Opening input file: " << inputFileName << std::endl;
    TFile file(inputFileName.c_str(), "READ");
    if (!file.IsOpen()) {
        std::cerr << "Error: Unable to open input file '" << inputFileName << "'\n";
        return;
    }

    // Create a new ROOT file for normalized histograms
    std::string outputFileName = outputDir + "/normalized_all.root";
    std::cout << "Creating output file: " << outputFileName << std::endl;
    TFile outputFile(outputFileName.c_str(), "RECREATE");
    if (!outputFile.IsOpen()) {
        std::cerr << "Error: Unable to create output file '" << outputFileName << "'\n";
        file.Close();
        return;
    }

    // Loop through each key in the file
    std::cout << "Looping through histograms..." << std::endl;
    TKey *key;
    TIter nextkey(file.GetListOfKeys());
    while ((key = (TKey*)nextkey())) {
        TObject *obj = key->ReadObj();
        if (obj->InheritsFrom("TH1")) { // Check if object is a histogram
            TH1* originalHistogram = dynamic_cast<TH1*>(obj);
            if (originalHistogram) {
                std::string histogramName = originalHistogram->GetName();
                long entries = originalHistogram->GetEntries();

                // Find the cross-section histogram
                TH1* CrossSectionHisto = dynamic_cast<TH1*>(file.Get("XSection"));
                if (CrossSectionHisto) {
                    CrossSection = CrossSectionHisto->GetEntries();
                } else {
                    std::cerr << "Warning: Cross section histogram not found!\n";
                    CrossSection = 1.0; // Default to 1.0 if not found
                }

                // Perform normalization run using the luminosity value and cross-section
                if (entries > 0) {
                    long unsigned scaling_factor = (1000*luminosity * CrossSection/ entries);
                    originalHistogram->Scale(scaling_factor);
                    originalHistogram->Scale(1e-14);
                    originalHistogram->Scale(1e-14);
                    originalHistogram->Scale(1e-14);
                    originalHistogram->Scale(1e-14);
                    originalHistogram->SetDirectory(&outputFile);
                    std::cout << "Normalized histogram '" << histogramName << "' with " << entries << " entries with scaling factor of: " << scaling_factor <<"*(1e-28)^2" "\n";
                } else {
                    std::cerr << "Error: Invalid entry count for histogram '" << histogramName << "'\n";
                }
            } else {
                std::cerr << "Warning: Histogram not found in input file\n";
            }
        }
    }

    // Write and close the output file
    std::cout << "Writing output file..." << std::endl;
    outputFile.Write();
    outputFile.Close();
    file.Close();
}

int main(int argc, char* argv[]) {
    if (argc != 3 ) {
        std::cerr << "Usage: " << argv[0] << "<input_file> <luminosity_constant>\n";
        return 1;
    }

   
    std::string inputFileName = argv[2];
    double luminosity = std::stod(argv[3]);
    std::string outputDir = "./MC/analyses/normalised";

    // Print out file paths for debugging
    
    std::cout << "Input File: " << inputFileName << std::endl;

    // Perform normalization run on histograms
    performNormalizationRun(inputFileName, luminosity, outputDir);

    return 0;
}

