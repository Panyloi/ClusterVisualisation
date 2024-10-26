#!/bin/bash

pythonbin=./.venv/bin/python3
script_path=./parallel_tests.py

echo "source_data;config_id;out_data;time" > ./pngs/results.csv

$pythonbin $script_path data/kamada_l1-mutual_attraction_2d.csv pngs/kamada_l1-mutual_attraction_2d_iter iterative & \
$pythonbin $script_path data/kk_swap_2d.csv pngs/kk_swap_2d_iter iterative & \
\
$pythonbin $script_path data/kamada_l1-mutual_attraction_2d.csv pngs/kamada_l1-mutual_attraction_2d_daq divide_and_conquare & \
$pythonbin $script_path data/kk_swap_2d.csv pngs/kk_swap_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/kamada_l1-mutual_attraction_2d.csv pngs/kamada_l1-mutual_attraction_2d_glob global & \
$pythonbin $script_path data/kk_swap_2d.csv pngs/kk_swap_2d_glob global & \
\
$pythonbin $script_path data/size-12/mds_ged_blp_2d.csv pngs/size-12/mds_ged_blp_2d_iter iterative & \
$pythonbin $script_path data/size-12/mds_katz_cen_2d.csv pngs/size-12/mds_katz_cen_2d_iter iterative & \
$pythonbin $script_path data/size-12/mds_sphr_2d.csv pngs/size-12/mds_sphr_2d_iter iterative & \
$pythonbin $script_path data/size-12/pca_katz_cen_2d.csv pngs/size-12/pca_katz_cen_2d_iter iterative & \
\
$pythonbin $script_path data/size-12/mds_ged_blp_2d.csv pngs/size-12/mds_ged_blp_2d_daq divide_and_conquare & \
$pythonbin $script_path data/size-12/mds_katz_cen_2d.csv pngs/size-12/mds_katz_cen_2d_daq divide_and_conquare & \
$pythonbin $script_path data/size-12/mds_sphr_2d.csv pngs/size-12/mds_sphr_2d_daq divide_and_conquare & \
$pythonbin $script_path data/size-12/pca_katz_cen_2d.csv pngs/size-12/pca_katz_cen_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/size-12/mds_ged_blp_2d.csv pngs/size-12/mds_ged_blp_2d_glob global & \
$pythonbin $script_path data/size-12/mds_katz_cen_2d.csv pngs/size-12/mds_katz_cen_2d_glob global & \
$pythonbin $script_path data/size-12/mds_sphr_2d.csv pngs/size-12/mds_sphr_2d_glob global & \
$pythonbin $script_path data/size-12/pca_katz_cen_2d.csv pngs/size-12/pca_katz_cen_2d_glob global & \
\
$pythonbin $script_path data/size-7/mds_ged_blp_2d.csv pngs/size-7/mds_ged_blp_2d_iter iterative & \
$pythonbin $script_path data/size-7/mds_katz_cen_2d.csv pngs/size-7/mds_katz_cen_2d_iter iterative & \
\
$pythonbin $script_path data/size-7/mds_ged_blp_2d.csv pngs/size-7/mds_ged_blp_2d_daq divide_and_conquare & \
$pythonbin $script_path data/size-7/mds_katz_cen_2d.csv pngs/size-7/mds_katz_cen_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/size-7/mds_ged_blp_2d.csv pngs/size-7/mds_ged_blp_2d_glob global & \
$pythonbin $script_path data/size-7/mds_katz_cen_2d.csv pngs/size-7/mds_katz_cen_2d_glob global & \
\
$pythonbin $script_path data/size-25/mds_katz_cen_2d.csv pngs/size-25/mds_katz_cen_2d_iter iterative & \
\
$pythonbin $script_path data/size-25/mds_katz_cen_2d.csv pngs/size-25/mds_katz_cen_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/size-25/mds_katz_cen_2d.csv pngs/size-25/mds_katz_cen_2d_glob global & \
\
$pythonbin $script_path data/ordinal-8/mds_ged_blp_2d.csv pngs/ordinal-8/mds_ged_blp_2d_iter iterative & \
$pythonbin $script_path data/ordinal-8/mds_katz_cen_2d.csv pngs/ordinal-8/mds_katz_cen_2d_iter iterative & \
\
$pythonbin $script_path data/ordinal-8/mds_ged_blp_2d.csv pngs/ordinal-8/mds_ged_blp_2d_daq divide_and_conquare & \
$pythonbin $script_path data/ordinal-8/mds_katz_cen_2d.csv pngs/ordinal-8/mds_katz_cen_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/ordinal-8/mds_ged_blp_2d.csv pngs/ordinal-8/mds_ged_blp_2d_glob global & \
$pythonbin $script_path data/ordinal-8/mds_katz_cen_2d.csv pngs/ordinal-8/mds_katz_cen_2d_glob global & \
\
$pythonbin $script_path data/move-12/mds_ged_blp_2d.csv pngs/move-12/mds_ged_blp_2d_iter iterative & \
$pythonbin $script_path data/move-12/mds_katz_cen_2d.csv pngs/move-12/mds_katz_cen_2d_iter iterative & \
\
$pythonbin $script_path data/move-12/mds_ged_blp_2d.csv pngs/move-12/mds_ged_blp_2d_daq divide_and_conquare & \
$pythonbin $script_path data/move-12/mds_katz_cen_2d.csv pngs/move-12/mds_katz_cen_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/move-12/mds_ged_blp_2d.csv pngs/move-12/mds_ged_blp_2d_glob global & \
$pythonbin $script_path data/move-12/mds_katz_cen_2d.csv pngs/move-12/mds_katz_cen_2d_glob global & \
