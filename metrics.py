"""This module contains functions for computing metrics on paths."""

from typing import Union
from shapely.geometry import LineString

def get_patch_of_line(line, width=0.5, resolution=16):
    if isinstance(line, list):
        line = LineString(line)
    # patch = line.buffer(width / 2.0, resolution=resolution, join_style='bevel', mitre_limit=1, cap_style='flat')
    patch = line.buffer(width / 2.0, resolution=resolution, join_style='round', mitre_limit=1, cap_style='round')
    return patch


def compute_area_of_path(line : Union[list, LineString]):
    if isinstance(line, list):
        line = LineString(line)
    patch = get_patch_of_line(line)
    return patch.area
