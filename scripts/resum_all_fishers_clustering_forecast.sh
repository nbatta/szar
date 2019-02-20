#!/usr/local/bin/bash

set -e

#Planck files
pl1="userdata/planck/savedFisher_szar_HighEllPlanck_fsky_0.2_mnuwwa_step_0.01.txt"
pl2="userdata/planck/savedFisher_szar_MidEllPlanck_fsky_0.4_mnuwwa_step_0.01.txt"
pl3="userdata/planck/savedFisher_szar_LowEllPlanck_fsky_0.6_mnuwwa_step_0.01.txt"

#SO base files
so_b_abund="userdata/so/base/so_v3_base_40_tsz_counts_no_cmb_no_desi_no_tau_prior_mnu_w_w0.txt"
so_b_clust="userdata/so/base/fisher_dc_SO-v3_base_40_owl2_v0.6_2019-02-05-21-18-27-EST.pkl"
so_b_clust_abias="userdata/so/base/fisher_dc_SO-v3_base_40_owl2_v0.6_2019-02-05-20-29-02-EST_abias.pkl"

#SO goal files
so_g_abund="userdata/so/goal/so_v3_goal_40_tsz_counts_no_cmb_no_desi_no_tau_prior_mnu_w_w0.txt"
so_g_clust="userdata/so/goal/fisher_dc_SO-v3_goal_40_owl2_v0.6_2019-02-05-21-32-31-EST.pkl"
so_g_clust_abias="userdata/so/goal/fisher_dc_SO-v3_goal_40_owl2_v0.6_2019-02-05-20-39-44-EST_abias.pkl"

#S4 files
s4_abund="userdata/s4/s4_v3style_40_tsz_counts_no_cmb_no_desi_no_tau_prior_mnu_w_w0.txt"
s4_clust="userdata/s4/fisher_dc_S4-1.0-CDT_grid-owl2_v0.6_2019-02-05-21-04-25-EST.pkl"
s4_clust_abias="userdata/s4/fisher_dc_S4-1.0-CDT_grid-owl2_v0.6_2019-02-05-17-41-47-EST_abias.pkl"

python bin/combine_fishers.py -o userdata/so/base/sum_so_base_abund_plus_planck.pkl $so_b_abund $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/base/sum_so_base_abund_plus_planck_clustering.pkl $so_b_abund $so_b_clust $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/base/sum_so_base_abund_plus_planck_clustering_abias.pkl $so_b_abund $so_b_clust_abias $pl1 $pl2 $pl3

python bin/combine_fishers.py -o userdata/so/goal/sum_so_goal_abund_plus_planck.pkl $so_g_abund $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/goal/sum_so_goal_abund_plus_planck_clustering.pkl $so_g_abund $so_g_clust $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/so/goal/sum_so_goal_abund_plus_planck_clustering_abias.pkl $so_g_abund $so_g_clust_abias $pl1 $pl2 $pl3

python bin/combine_fishers.py -o userdata/s4/sum_s4_abund_plus_planck.pkl $s4_abund $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/s4/sum_s4_abund_plus_planck_clustering.pkl $s4_abund $s4_clust $pl1 $pl2 $pl3
python bin/combine_fishers.py -o userdata/s4/sum_s4_abund_plus_planck_clustering_abias.pkl $s4_abund $s4_clust_abias $pl1 $pl2 $pl3
