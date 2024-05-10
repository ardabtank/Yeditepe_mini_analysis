#!/bin/bash

directory="./"
non_exotic_channelNumbers="410000 410012 410011 410014 410013 410026 410025 361106 361107 361108 361100 361101 361102 361103 361104 361105 364156 364157 364158 364159 364160 364161 364162 364163 364164 364165 364166 364167 364168 364169 364170 364171 364172 364173 364174 364175 364176 364177 364178 364179 364180 364181 364182 364183 364184 364185 364186 364187 364188 364189 364190 364191 364192 364193 364194 364195 364196 364197 364100 364101 364102 364103 364104 364105 364106 364107 364108 364109 364110 364111 364112 364113 364114 364115 364116 364117 364118 364119 364120 364121 364122 364123 364124 364125 364126 364127 364128 364129 364130 364131 364132 364133 364134 364135 364136 364137 364138 364139 364140 364141 363359 363360 363492 363356 363490 363358 363489 363491 363493 345324 345323 345060 344235 341947 341964 343981 345041 345318 345319 341081"

# Convert it to an array.
IFS=' ' read -r -a non_exotic_array <<< "$non_exotic_channelNumbers"

# Check the files in MC
for file in "$directory"/*; do
    if [ -f "$file" ]; then
        numberFound=false
        for number in "${non_exotic_array[@]}"; do
            if [[ "$file" =~ $number ]]; then
                numberFound=true
                break
            fi
        done

        if [ "$numberFound" = true ]; then
            # Remove if the relevant numbers are not included in the name of file.
            echo "$file is removed because it has no any number of non-exotic particles."
            rm "$file"
        else
            echo "$file is not removed because it has one of the non-exotic particle numbers."
        fi
    fi
done

echo "Exotic particles have been removed."

