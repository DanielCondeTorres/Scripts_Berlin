#!/usr/bin/env python3
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate and plot RDF panels for CG beads')
    parser.add_argument('-pdb', '--pdb', required=True, help='Structure file (GRO/PDB)')
    parser.add_argument('-xtc', '--xtc', required=True, help='Trajectory file (XTC/DCD)')
    parser.add_argument('-skip', '--skip', type=int, default=1, help='Frame skip for faster analysis')
    parser.add_argument('-sel', '--selection', default='all', help='Overall atom selection')
    parser.add_argument('-nbins', '--nbins', type=int, default=200, help='Number of RDF bins')
    parser.add_argument('-range', '--range', nargs=2, type=float, default=[1.0, 20.0], help='Distance range in Å')
    parser.add_argument('-cols', '--cols', type=int, default=3, help='Number of columns in panel')
    return parser.parse_args()

def calculate_rdf_matrix(u, selections, rdf_range, nbins):
    """Compute RDFs for all bead pairs and return data dict"""
    rdf_data = {}
    for sel1 in selections:
        rdf_data[sel1] = {}
        g1 = u.select_atoms(sel1)
        if len(g1)==0: continue
        for sel2 in selections:
            g2 = u.select_atoms(sel2)
            if len(g2)==0: continue
            rdf = InterRDF(g1, g2, nbins=nbins, range=rdf_range, norm='density')
            rdf.run()
            rdf_data[sel1][sel2] = (rdf.bins, rdf.rdf)
            # save each text file if desired
            np.savetxt(f"rdf_{sel1.replace(' ', '')}_{sel2.replace(' ', '')}.txt",
                       np.column_stack((rdf.bins, rdf.rdf)))
    return rdf_data

def plot_panel(rdf_data, selections, cols, output='rdf_panel.png'):
    """Create panel: one subplot per sel1 showing all sel1-sel2 curves"""
    n = len(selections)
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    for idx, sel1 in enumerate(selections):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.set_title(f'RDFs for {sel1}')
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('g(r)')
        ax.grid(alpha=0.3)
        for sel2 in selections:
            if sel2 in rdf_data.get(sel1, {}):
                bins, g = rdf_data[sel1][sel2]
                alpha = 1.0 if sel1==sel2 else 0.5
                lw = 2.0 if sel1==sel2 else 1.5
                ax.plot(bins, g, label=sel2, alpha=alpha, linewidth=lw)
        ax.legend(fontsize=8)
    # hide unused
    for idx in range(n, rows*cols):
        r = idx // cols; c = idx % cols
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved panel to {output}")


def main():
    args = parse_arguments()
    u = mda.Universe(args.pdb, args.xtc)
    if args.skip>1:
        u.trajectory[::args.skip]
    all_sel = u.select_atoms(args.selection)
    bead_names = sorted(set(all_sel.names))
    selections = [f'name {b}' for b in bead_names]
    print(f"Bead types: {bead_names}")
    rdf_range = tuple(args.range)
    rdf_data = calculate_rdf_matrix(u, selections, rdf_range, args.nbins)
    plot_panel(rdf_data, selections, args.cols)

if __name__=='__main__':
    main()
