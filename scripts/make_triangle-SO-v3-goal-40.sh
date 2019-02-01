#!/usr/local/bin/bash

set -e

derivs="datatest/SO-v3-goal-40_grid-owl2_v0.6_fish_derivs_$1.npy"
factors="datatest/SO-v3-goal-40_grid-owl2_v0.6_fish_factor_$1.npy"
params="datatest/SO-v3-goal-40_grid-owl2_v0.6_params_$1.npy"
extrafishers=$2
triplot_file="figs/SO-v3-goal-40-owl2_v0.6_$1.png"

python bin/make_fisher_clust.py -d "$derivs" -f "$factors" -p "$params" --extra-fishers "$extrafishers" --make-tri "$triplot_file"
