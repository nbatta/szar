#!/usr/local/bin/bash

set -e

derivs="userdata/so/prefisher/SO-v3-base-40_grid-owl2_v0.6_fish_derivs_$1.npy"
factors="userdata/so/prefisher/SO-v3-base-40_grid-owl2_v0.6_fish_factor_$1.npy"
params="userdata/so/prefisher/SO-v3-base-40_grid-owl2_v0.6_params_$1.npy"
outfile="dc_SO-v3_base_40_owl2_v0.6_$1"

python bin/make_fisher_clust.py -o "$outfile" -d "$derivs" -f "$factors" -p "$params" 
