#!/usr/local/bin/bash

set -e

DIR="userdata/s4/prefisher"

FIGNAME="$1"

UP="$DIR/S4-1.0-CDT_grid-owl2_v0.6_psups_2019-02-05-21-04-25-EST.npy"
DOWN="$DIR/S4-1.0-CDT_grid-owl2_v0.6_psdowns_2019-02-05-21-04-25-EST.npy"
PREF="$DIR/S4-1.0-CDT_grid-owl2_v0.6_fish_factor_2019-02-05-21-04-25-EST.npy"
PARAM="$DIR/S4-1.0-CDT_grid-owl2_v0.6_params_2019-02-05-21-04-25-EST.npy"

python bin/plotClusterSpectra.py $FIGNAME -u $UP -d $DOWN -p $PREF -par $PARAM
