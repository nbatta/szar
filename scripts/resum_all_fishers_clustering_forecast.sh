#!/usr/local/bin/bash

set -e

#Planck files
pl1="userdata/planck/savedFisher_szar_HighEllPlanck_fsky_0.2_mnuwwa_step_0.01.txt"
pl2="userdata/planck/savedFisher_szar_MidEllPlanck_fsky_0.4_mnuwwa_step_0.01.txt"
pl3="userdata/planck/savedFisher_szar_LowEllPlanck_fsky_0.6_mnuwwa_step_0.01.txt"

#SO base files
so_b_abund="userdata/so/base/savedFisher_SO-v3-base-40_grid-owl2_owl2_v1.1_no_planck.pkl"
so_b_clust="userdata/so/base/fisher_dc_SO-v3_base_40_owl2_v1.1_2019-03-22-14-02-36-EDT.pkl"
so_b_clust_abias="userdata/so/base/fisher_dc_SO-v3_base_40_owl2_v1.1_2019-03-22-11-59-14-EDT_abias.pkl"

#SO goal files
so_g_abund="userdata/so/goal/savedFisher_SO-v3-goal-40_grid-owl2_owl2_v1.1_no_planck.pkl"
so_g_clust="userdata/so/goal/fisher_dc_SO-v3_goal_40_owl2_v1.1_2019-03-22-14-11-22-EDT.pkl"
so_g_clust_abias="userdata/so/goal/fisher_dc_SO-v3_goal_40_owl2_v1.1_2019-03-22-12-28-15-EDT_abias.pkl"

#S4 files
s4_abund="userdata/s4/savedFisher_S4-1.0-CDT_grid-owl2_owl2_v1.1_no_planck.pkl"
s4_clust="userdata/s4/fisher_dc_S4-1.0-CDT_grid-owl2_v1.1_2019-03-22-13-52-58-EDT.pkl"
s4_clust_abias="userdata/s4/fisher_dc_S4-1.0-CDT_grid-owl2_v1.1_2019-03-22-12-36-44-EDT_abias.pkl"

python bin/combine_fishers.py -o userdata/so/base/sum_so_base_abund_plus_planck.pkl $so_b_abund $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/base/sum_so_base_abund_plus_planck_clustering.pkl $so_b_abund $so_b_clust $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/base/sum_so_base_abund_plus_planck_clustering_abias.pkl $so_b_abund $so_b_clust_abias $pl1 $pl2 $pl3

python bin/combine_fishers.py -o userdata/so/goal/sum_so_goal_abund_plus_planck.pkl $so_g_abund $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/goal/sum_so_goal_abund_plus_planck_clustering.pkl $so_g_abund $so_g_clust $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/goal/sum_so_goal_abund_plus_planck_clustering_abias.pkl $so_g_abund $so_g_clust_abias $pl1 $pl2 $pl3

python bin/combine_fishers.py -o userdata/s4/sum_s4_abund_plus_planck.pkl $s4_abund $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/s4/sum_s4_abund_plus_planck_clustering.pkl $s4_abund $s4_clust $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/s4/sum_s4_abund_plus_planck_clustering_abias.pkl $s4_abund $s4_clust_abias $pl1 $pl2 $pl3
