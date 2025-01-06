from src.main import draw_maps, draw_maps_editor

# open editor make generated from mds_katz_cen_2d.csv dataset.
draw_maps_editor("data/size-7/mds_katz_cen_2d.csv")

# generate .png of mds_katz_cen_2d.csv dataset with iterative label generation method
draw_maps("data/size-7/mds_katz_cen_2d.csv", "mds_katz_cen_2d_daq", config_id='iterative')
