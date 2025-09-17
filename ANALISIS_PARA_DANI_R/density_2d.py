
#!/usr/bin/env python3
"""
Script to compute the Z-density profile of lipids (heads vs tails),
centered on the membrane COM.

Usage:
    python density_z.py -pdb structure.pdb -dcd trajectory.dcd
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
    return lipids, heads, tails

def calculate_density_z(universe, lipids, heads, tails, bins=100, start=None, stop=None, step=None):
    """Compute normalized Z-density profile centered at membrane COM"""
    traj = universe.trajectory
    n_frames = len(traj)

    start = start if start is not None else 0
    stop = stop if stop is not None else n_frames
    step = step if step is not None else 1

    all_heads_z = []
    all_tails_z = []

    for ts in traj[start:stop:step]:
        # Center Z coordinates at membrane COM
        com_z = lipids.center_of_mass()[2]
        heads_z = heads.positions[:, 2] - com_z
        tails_z = tails.positions[:, 2] - com_z

        all_heads_z.extend(heads_z)
        all_tails_z.extend(tails_z)

    # Define range based on min/max after centering
    z_min = min(np.min(all_heads_z), np.min(all_tails_z))
    z_max = max(np.max(all_heads_z), np.max(all_tails_z))

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

    plt.xlabel("Z (Å) relative to membrane COM", fontsize=16)
    plt.ylabel("Normalized density", fontsize=16)
    plt.title("Z-density profile of lipids (centered)", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output}_densityZ_centered.png", dpi=300)
    plt.show()

def main():
    args = parse_args()

    if not os.path.exists(args.pdb):
        print(f"Error: {args.pdb} does not exist")
        sys.exit(1)
    if not os.path.exists(args.dcd):
        print(f"Error: {args.dcd} does not exist")
        sys.exit(1)

    print(f"Loading universe: {args.pdb}, {args.dcd}")
    universe = mda.Universe(args.pdb, args.dcd)
    print(f"Atoms: {len(universe.atoms)}, Frames: {len(universe.trajectory)}")

    lipids, heads, tails = detect_groups(universe, args.sel)

    z_centers, H_heads, H_tails = calculate_density_z(
        universe, lipids, heads, tails, bins=args.bins,
        start=args.start, stop=args.stop, step=args.step
    )

    np.savetxt(f"{args.o}_densityZ_heads_centered.txt", np.column_stack((z_centers, H_heads)),
               header="Z (Å) relative to COM\tNormalized density (heads)")
    np.savetxt(f"{args.o}_densityZ_tails_centered.txt", np.column_stack((z_centers, H_tails)),
               header="Z (Å) relative to COM\tNormalized density (tails)")

    plot_density_z(z_centers, H_heads, H_tails, args.o)

    print("\n=== RESULTS ===")
    print(f"- {args.o}_densityZ_heads_centered.txt")
    print(f"- {args.o}_densityZ_tails_centered.txt")
    print(f"- {args.o}_densityZ_centered.png")

if __name__ == "__main__":
    main()
