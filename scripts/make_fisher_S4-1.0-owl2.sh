#!/usr/local/bin/bash

set -e

derivs="userdata/s4/prefisher/S4-1.0-CDT_grid-owl2_v1.1_fish_derivs_$1.npy"
factors="userdata/s4/prefisher/S4-1.0-CDT_grid-owl2_v1.1_fish_factor_$1.npy"
params="userdata/s4/prefisher/S4-1.0-CDT_grid-owl2_v1.1_params_$1.npy"
outfile="dc_S4-1.0-CDT_grid-owl2_v1.1_$1"
maxkh="0.14"
inifile="input/pipeline.ini"
exp="S4-1.0-CDT"
grid="grid-owl2"

python bin/make_fisher_clust.py -o "$outfile" -d "$derivs" -f "$factors" -p "$params" --maxkh "$maxkh" --inifile "$inifile" --expname "$exp" --gridname "$grid"
