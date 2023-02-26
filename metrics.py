"""This module contains functions for computing metrics on paths."""

from typing import Union
from shapely.geometry import LineString

def get_patch_of_line(line):
    if isinstance(line, list):
        line = LineString(line)
    patch = line.buffer(4, resolution=16, join_style='round', mitre_limit=1)
    return patch


def compute_area_of_path(line : Union[list, LineString]):
    if isinstance(line, list):
        line = LineString(line)
    patch = get_patch_of_line(line)
    return patch.area
