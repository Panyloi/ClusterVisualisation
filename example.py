from src.main import draw_maps, draw_maps_editor

import time
# draw_maps_editor("data/kk_swap_2d.csv", None)
s = time.time()
draw_maps("data/smaller_data.csv", "testgen", config_id='iterative')
e = time.time()
print(f"{e-s}s")
