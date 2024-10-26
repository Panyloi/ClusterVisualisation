from src.main import draw_maps, draw_maps_editor

import time
import argparse
import fcntl

parser = argparse.ArgumentParser()
parser.add_argument("source_data", type=str, help="csv file path with data")
parser.add_argument("out_path", type=str, help="path where to save output")
parser.add_argument("config", type=str, help="config id")

args = parser.parse_args()

s = time.time()
draw_maps(args.source_data, args.out_path, config_id=args.config)
e = time.time()
with open("pngs/results.csv", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(f"{args.source_data};{args.config};{args.out_path};{e-s}\n")
    fcntl.flock(f, fcntl.LOCK_UN)
print(f"{e-s}")
