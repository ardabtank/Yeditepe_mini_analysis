#include <iostream>
#include <filesystem>
#include <string.h>
#include "TTree.h"
#include "TFile.h"
#include "tree_reader.h"


void analyse_single_file(const char* infile, const char* destination){
    std::cout<<"Opening root file : "<<infile<<"\n";
    if( ! std::filesystem::exists(infile) ){
        throw std::runtime_error("input file does not exist! ");
    }
    if(std::filesystem::exists(destination)) {
        std::cout<<"Output : "<<destination<<" already exists, going to override it\n"; 
    }

    TFile* in = TFile::Open(infile, "READ");
    TTree* input_tree = in->Get<TTree>("mini");
    if(input_tree == nullptr) {
        throw std::runtime_error("Unable to read tree!");
    }

    tree_reader myreader{input_tree};
    myreader.setDestination(destination);
    myreader.Loop();
}



int main(int argc , char* argv[]){
    if(argc != 3) {
        std::cout<<"Commandline error. Usage : \n"<<argv[0]<<" <input.root> <output.root>\n";
        throw std::runtime_error("cl error");
    }
    if(strcmp(argv[1],argv[2]) == 0) {
        throw std::runtime_error("Input and output files cannot be the same!");
    }

    analyse_single_file(argv[1],argv[2]);
    
    std::cout<<argv[1]<<" : DONE\n";

    std::cout<<argv[2]<<" is created\n";

    return 0;
}