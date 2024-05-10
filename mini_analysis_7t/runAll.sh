#!/bin/bash

indir=$1

for f in $(ls $indir | grep root);do

fullpath=$(readlink -e $indir)/$f
runnner.exe "$fullpath" "out_${f}"

done
