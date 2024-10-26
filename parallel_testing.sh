#!/bin/bash

pythonbin=./.venv/bin/python3
script_path=./parallel_tests.py

$pythonbin $script_path data/kamada_l1-mutual_attraction_2d.csv pngs/kamada_l1-mutual_attraction_2d_iter iterative & \
$pythonbin $script_path data/kk_swap_2d.csv pngs/kk_swap_2d_iter iterative & \
\
$pythonbin $script_path data/kamada_l1-mutual_attraction_2d.csv pngs/kamada_l1-mutual_attraction_2d_daq divide_and_conquare & \
$pythonbin $script_path data/kk_swap_2d.csv pngs/kk_swap_2d_daq divide_and_conquare & \
\
$pythonbin $script_path data/size-12/mds_ged_blp_2d.csv pngs/size-12/mds_ged_blp_2d_iter iterative & \
$pythonbin $script_path data/size-12/mds_katz_cen_2d.csv pngs/size-12/mds_katz_cen_2d_iter iterative & \
$pythonbin $script_path data/size-12/mds_sph_2d.csv pngs/size-12/mds_sph_2d_iter iterative & \
$pythonbin $script_path data/size-12/mds_sphr_2d.csv pngs/size-12/mds_sphr_2d_iter iterative & \
$pythonbin $script_path data/size-12/pca_katz_cen_2d.csv pngs/size-12/pca_katz_cen_2d_iter iterative