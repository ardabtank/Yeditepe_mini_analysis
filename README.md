a simple readme

to compile the paint.c : g++ -o ressam paint.c -std=c++11 -Wall `root-config --cflags --libs`

in the dir that you are working on simply just write runAll.sh {dir_of_data_or_mc} to take the analysis of Data or MC

once compiled, ressam takes two arguments; the usage is = ressam {dir}out_data_all.root {dir}out_mc_all.root
red one is always the second histogram given to the ressam
