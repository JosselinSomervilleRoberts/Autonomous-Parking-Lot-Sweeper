"""This module contains functions for computing metrics on paths."""

def getPatchOfLine(line):
    patch = line.buffer(0.1, resolution=16, join_style='round', cap_style='flat', mitre_limit=1)
    return patch

def computAreaOfPath(line):
    patch = getPatchOfLine(line)
    return patch.area