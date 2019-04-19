#!/usr/local/bin/bash

set -e

derivs="userdata/so/prefisher/SO-v3-goal-40_grid-owl2_v1.1_fish_derivs_$1.npy"
factors="userdata/so/prefisher/SO-v3-goal-40_grid-owl2_v1.1_fish_factor_$1.npy"
params="userdata/so/prefisher/SO-v3-goal-40_grid-owl2_v1.1_params_$1.npy"
outfile="dc_SO-v3_goal_40_owl2_v1.1_$1"
maxkh="0.14"
inifile="input/pipeline.ini"
exp="SO-v3-goal-40"
grid="grid-owl2"

python bin/make_fisher_clust.py -o "$outfile" -d "$derivs" -f "$factors" -p "$params" --maxkh "$maxkh" --inifile "$inifile" --expname "$exp" --gridname "$grid"
