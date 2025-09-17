#!/usr/bin/env python3
"""
Script para calcular el perfil de densidad 2D de lípidos (cabezas vs colas).

Uso:
    python lipid_density_2d.py -pdb estructura.pdb -dcd trayectoria.dcd

Salida:
    - Mapas de densidad 2D (colas en morado, cabezas en rojo)
    - Archivos de texto con los histogramas
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

# Posibles nombres de átomos de cabeza (headgroups)
HEADGROUP_NAMES = [
    "PO4", "P", "ROH", "NH3", "CNO", "GL1", "GL2", "O.*", "N.*"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Perfil de densidad 2D de lípidos")
    parser.add_argument("-pdb", required=True, help="Archivo PDB")
    parser.add_argument("-dcd", required=True, help="Archivo DCD")
    parser.add_argument("-sel", default="resname DPPC DOPC POPC", help="Selección de lípidos")
    parser.add_argument("-start", type=int, default=None, help="Frame inicial")
    parser.add_argument("-stop", type=int, default=None, help="Frame final")
    parser.add_argument("-step", type=int, default=None, help="Paso entre frames")
    parser.add_argument("-o", default="lipid_density2D", help="Prefijo de salida")
    parser.add_argument("-bins", type=int, default=100, help="Número de bins en XY")
    return parser.parse_args()

def detect_groups(universe, lipid_sel):
    """Detecta automáticamente átomos de cabeza y de cola en los lípidos"""
    lipids = universe.select_atoms(lipid_sel)
    if len(lipids) == 0:
        print(f"Error: no se encontraron lípidos con selección '{lipid_sel}'")
        sys.exit(1)

    # Headgroups = átomos que matchean nombres conocidos
    head_sel_query = " or ".join([f"name {name}" for name in HEADGROUP_NAMES])
    heads = lipids.select_atoms(head_sel_query)

    # Colas = el resto de átomos de los lípidos
    tails = lipids.difference(heads)

    print(f"Lípidos: {len(lipids.residues)}");
    print(f"Átomos de cabeza: {len(heads)}")
    print(f"Átomos de cola: {len(tails)}")
    return heads, tails

def calculate_density(universe, heads, tails, bins=100, start=None, stop=None, step=None):
    """Calcula histogramas 2D de densidad para cabezas y colas"""
    traj = universe.trajectory
    n_frames = len(traj)

    start = start if start is not None else 0
    stop = stop if stop is not None else n_frames
    step = step if step is not None else 1

    heads_hist = None
    tails_hist = None

    for i, ts in enumerate(traj[start:stop:step]):
        # Coordenadas XY
        h_xy = heads.positions[:, :2]
        t_xy = tails.positions[:, :2]

        # Límites de la caja
        Lx, Ly = ts.dimensions[0], ts.dimensions[1]

        # Histograma 2D
        Hh, xedges, yedges = np.histogram2d(h_xy[:,0], h_xy[:,1], bins=bins, range=[[0,Lx],[0,Ly]])
        Ht, _, _ = np.histogram2d(t_xy[:,0], t_xy[:,1], bins=bins, range=[[0,Lx],[0,Ly]])

        if heads_hist is None:
            heads_hist = Hh
            tails_hist = Ht
        else:
            heads_hist += Hh
            tails_hist += Ht

    # Normalizar por número de frames
    heads_hist = heads_hist / ((stop-start)//step)
    tails_hist = tails_hist / ((stop-start)//step)

    return heads_hist, tails_hist, xedges, yedges

def plot_density(heads_hist, tails_hist, xedges, yedges, output):
    """Genera la figura con cabezas en rojo y colas en morado"""
    plt.figure(figsize=(10,8))

    # Transponer para que coincidan los ejes
    H_heads = heads_hist.T
    H_tails = tails_hist.T

    # Crear imagen combinada (RGB)
    # Normalización por intensidad máxima
    H_heads /= H_heads.max() if H_heads.max() > 0 else 1
    H_tails /= H_tails.max() if H_tails.max() > 0 else 1

    rgb_image = np.zeros((H_heads.shape[0], H_heads.shape[1], 3))
    rgb_image[:,:,0] = H_heads   # Rojo = cabezas
    rgb_image[:,:,2] = H_tails   # Azul = colas → morado

    plt.imshow(rgb_image, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='equal')
    plt.xlabel("X (Å)")
    plt.ylabel("Y (Å)")
    plt.title("Perfil de densidad 2D de lípidos (rojo=cabezas, morado=colas)")
    plt.colorbar(plt.cm.ScalarMappable(cmap="Reds"), label="Cabezas (intensidad relativa)", fraction=0.046)
    plt.colorbar(plt.cm.ScalarMappable(cmap="Purples"), label="Colas (intensidad relativa)", fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{output}_density2D.png", dpi=300)
    plt.show()

def main():
    args = parse_args()

    # Validar archivos
    if not os.path.exists(args.pdb):
        print(f"Error: {args.pdb} no existe")
        sys.exit(1)
    if not os.path.exists(args.dcd):
        print(f"Error: {args.dcd} no existe")
        sys.exit(1)

    # Cargar universo
    print(f"Cargando universo: {args.pdb}, {args.dcd}")
    universe = mda.Universe(args.pdb, args.dcd)
    print(f"Átomos: {len(universe.atoms)}, Frames: {len(universe.trajectory)}")

    # Detectar cabezas y colas
    heads, tails = detect_groups(universe, args.sel)

    # Calcular densidad
    heads_hist, tails_hist, xedges, yedges = calculate_density(
        universe, heads, tails, bins=args.bins,
        start=args.start, stop=args.stop, step=args.step
    )

    # Guardar histogramas
    np.savetxt(f"{args.o}_heads_hist.txt", heads_hist, fmt="%.6f")
    np.savetxt(f"{args.o}_tails_hist.txt", tails_hist, fmt="%.6f")

    # Graficar
    plot_density(heads_hist, tails_hist, xedges, yedges, args.o)

    print("\n=== RESULTADOS ===")
    print(f"Archivos guardados:")
    print(f"- {args.o}_heads_hist.txt")
    print(f"- {args.o}_tails_hist.txt")
    print(f"- {args.o}_density2D.png")

if __name__ == "__main__":
    main()
