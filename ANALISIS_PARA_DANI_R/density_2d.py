#!/usr/bin/env python3
"""
Script to compute the Z-density profile of lipids (heads vs tails).

Usage:
    python density_z.py -pdb structure.pdb -dcd trajectory.dcd

Output:
    - densityZ_heads.txt (normalized profile of head atoms)
    - densityZ_tails.txt (normalized profile of tail atoms)
    - densityZ.png (figure with Z-density profile)
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

# Possible atom names in lipid headgroups
HEADGROUP_NAMES = [
    "PO4", "P", "ROH", "NH3", "CNO", "GL1", "GL2", "O.*", "N.*"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Z-density profile of lipids")
    parser.add_argument("-pdb", required=True, help="PDB structure file")
    parser.add_argument("-dcd", required=True, help="DCD trajectory file")
    parser.add_argument("-sel", default="resname DPPC DOPC POPC", help="Lipid selection")
    parser.add_argument("-start", type=int, default=None, help="First frame to analyze")
    parser.add_argument("-stop", type=int, default=None, help="Last frame to analyze")
    parser.add_argument("-step", type=int, default=None, help="Step between frames")
    parser.add_argument("-o", default="lipid_densityZ", help="Output prefix")
    parser.add_argument("-bins", type=int, default=100, help="Number of bins along Z")
    return parser.parse_args()

def detect_groups(universe, lipid_sel):
    """Detect head and tail atoms automatically in lipids"""
    lipids = universe.select_atoms(lipid_sel)
    if len(lipids) == 0:
        print(f"Error: no lipids found with selection '{lipid_sel}'")
        sys.exit(1)

    head_sel_query = " or ".join([f"name {name}" for name in HEADGROUP_NAMES])
    heads = lipids.select_atoms(head_sel_query)
    tails = lipids.difference(heads)

    print(f"Lipids detected: {len(lipids.residues)}")
    print(f"Head atoms: {len(heads)}")
    print(f"Tail atoms: {len(tails)}")
    return heads, tails

def calculate_density_z(universe, heads, tails, bins=100, start=None, stop=None, step=None):
    """Compute normalized Z-density profile"""
    traj = universe.trajectory
    n_frames = len(traj)

    start = start if start is not None else 0
    stop = stop if stop is not None else n_frames
    step = step if step is not None else 1

    all_heads_z = []
    all_tails_z = []

    for ts in traj[start:stop:step]:
        all_heads_z.extend(heads.positions[:, 2])
        all_tails_z.extend(tails.positions[:, 2])

    z_min, z_max = 0, ts.dimensions[2]

    H_heads, edges = np.histogram(all_heads_z, bins=bins, range=(z_min, z_max), density=True)
    H_tails, _     = np.histogram(all_tails_z, bins=bins, range=(z_min, z_max), density=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, H_heads, H_tails

def plot_density_z(z_centers, H_heads, H_tails, output):
    """Plot Z-density profile for head and tail atoms"""
    plt.figure(figsize=(8, 6))
    plt.plot(z_centers, H_heads, 'r-', lw=2, label="Heads")
    plt.plot(z_centers, H_tails, color="purple", lw=2, label="Tails")
    plt.fill_between(z_centers, H_heads, color="red", alpha=0.3)
    plt.fill_between(z_centers, H_tails, color="purple", alpha=0.3)

    plt.xlabel("Z height (Å)", fontsize=16)
    plt.ylabel("Normalized density", fontsize=16)
    plt.title("Z-density profile of lipids", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
