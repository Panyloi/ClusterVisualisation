from src.main import draw_maps, draw_maps_editor

import time
# draw_maps_editor("data/kk_swap_2d.csv", None)
s = time.time()
# draw_maps("data/kk_swap_2d.csv", "pngs/kk_swap_2d_iter", config_id='iterative')
# draw_maps("data/smaller_data.csv", "pngs/smaller_data_iter", config_id='iterative')
draw_maps("data/kamada_l1-mutual_attraction_2d.csv", "pngs/kamada_l1-mutual_attraction_2d_iter", config_id='iterative')
e = time.time()
print(f"{e-s}s")

s = time.time()
# draw_maps("data/kk_swap_2d.csv", "pngs/kk_swap_2d_daq", config_id='divide_and_conquare')
# draw_maps("data/smaller_data.csv", "pngs/smaller_data_daq", config_id='divide_and_conquare')
draw_maps("data/kamada_l1-mutual_attraction_2d.csv", "pngs/kamada_l1-mutual_attraction_2d_daq", config_id='divide_and_conquare')
e = time.time()
print(f"{e-s}s")
