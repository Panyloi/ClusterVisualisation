from src.main import draw_maps, draw_maps_editor

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source_data", type=str, help="csv file path with data")
parser.add_argument("out_path", type=str, help="path where to save output")
parser.add_argument("config", type=str, help="config id")

args = parser.parse_args()

s = time.time()
draw_maps(args.source_data, args.out_path, config_id=args.config)
e = time.time()
print(f"{e-s}")
