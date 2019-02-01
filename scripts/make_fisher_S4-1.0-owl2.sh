#!/usr/local/bin/bash

set -e

derivs="datatest/S4-1.0-CDT_grid-owl2_v0.6_fish_derivs_$1.npy"
factors="datatest/S4-1.0-CDT_grid-owl2_v0.6_fish_factor_$1.npy"
params="datatest/S4-1.0-CDT_grid-owl2_v0.6_params_$1.npy"
outfile="dc_S4-1.0-CDT_grid-owl2_v0.6_$1"

python bin/make_fisher_clust.py -o "$outfile" -d "$derivs" -f "$factors" -p "$params" 
