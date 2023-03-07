import os
import numpy as np
import random
from map import Map
from config import RenderOptions
from tqdm import tqdm

# Changes the values of the map to the values of the new map
mapping = {
    0: 0,
    1: 255,
    2: 64
}

len_maps = len(os.listdir("maps/"))
if len_maps == 0:
    print("No maps found in maps/ folder")
else:
    for i in tqdm(range(len_maps)):
        map = Map(100, 100, RenderOptions())
        map.load(f"maps/map_{i}.npy")
        for key in mapping:
            map.grid[map.grid == key] = mapping[key]
        map.save(f"maps/map_{i}")