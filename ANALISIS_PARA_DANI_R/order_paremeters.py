#!/usr/bin/env python3
"""
CG Membrane Analysis Script
Performs:
 1. Lipid tail order parameter (P2)
 2. Tail tilt distribution
 3. Membrane undulation spectrum
 4. Voronoi tessellation area distribution
 5. Void (packing defect) frequency
Generates data files and English-language plots in the output directory.
Usage: membrane_analysis.py -p top.pdb -x traj.xtc -o results/
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist
from scipy.fft import fft, fftfreq

# 1. Tail order parameter P2
def compute_order_parameter(universe, head_bead, tail_bead):
    """Calculate tail order parameter P2 for each frame."""
    times = []
    p2_vals = []
    for ts in ProgressBar(universe.trajectory):
        vals = []
        for res in universe.select_atoms('resname POPC').residues:
            head = res.atoms.select_atoms(f'name {head_bead}')
            tail = res.atoms.select_atoms(f'name {tail_bead}')
            if len(head) == 1 and len(tail) == 1:
                v = tail.positions[0] - head.positions[0]
                norm = np.linalg.norm(v)
                if norm > 0:
                    cos = v[2] / norm
                    vals.append(0.5 * (3 * cos**2 - 1))
        times.append(ts.time)
        p2_vals.append(np.mean(vals) if vals else np.nan)
    return np.array(times), np.array(p2_vals)

# 2. Tilt distribution
def compute_tilt_distribution(universe, head_bead, tail_bead, bins=50):
    """Calculate tilt angle distribution for lipid tails."""
    thetas = []
    for ts in ProgressBar(universe.trajectory):
        for res in universe.select_atoms('resname POPC').residues:
            head = res.atoms.select_atoms(f'name {head_bead}')
            tail = res.atoms.select_atoms(f'name {tail_bead}')
            if len(head) == 1 and len(tail) == 1:
                v = tail.positions[0] - head.positions[0]
                norm = np.linalg.norm(v)
                if norm > 0:
                    theta = np.degrees(np.arccos(np.clip(v[2] / norm, -1.0, 1.0)))
                    thetas.append(theta)
    thetas = np.array(thetas)
    thetas = thetas[np.isfinite(thetas)]
    hist, edges = np.histogram(thetas, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

# 3. Undulation spectrum
def compute_undulation_spectrum(universe, head_sel, frame_interval=1):
    height_list = []
    for ts in universe.trajectory[::frame_interval]:
        head = universe.select_atoms(head_sel)
        height_list.append(head.positions[:, 2])
    heights = np.array(height_list)
    heights -= np.nanmean(heights, axis=0)
    fft_vals = fft(heights, axis=0)
    power = np.abs(fft_vals) ** 2
    N = heights.shape[0]
    dt = frame_interval * universe.trajectory.dt
    freqs = fftfreq(N, d=dt)
    half = slice(1, N // 2)
    return freqs[half], np.nanmean(power[half, :], axis=1)

# 4. Voronoi area distribution
def compute_voronoi_areas(universe, head_sel):
    head = universe.select_atoms(head_sel)
    pts = head.positions[:, :2]
    vor = Voronoi(pts)
    areas = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if not region or -1 in region:
            areas.append(np.nan)
            continue
        poly = np.array([vor.vertices[i] for i in region])
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)
    return np.array(areas)

# 5. Void frequency (packing defects)
def compute_void_frequency(universe, head_sel, threshold=0.5):
    times = []
    freq = []
    for ts in universe.trajectory:
        head = universe.select_atoms(head_sel)
        coords = head.positions[:, :2]
        dists = pdist(coords)
        voids = np.sum(dists > threshold)
        times.append(ts.time)
        freq.append(voids)
    return np.array(times), np.array(freq)

# Plotting utilities
def save_line_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_histogram(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.bar(x, y, width=x[1] - x[0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pdb', required=True)
    parser.add_argument('-x', '--xtc', required=True)
    parser.add_argument('-o', '--outdir', default='results')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    u = mda.Universe(args.pdb, args.xtc)

    # Define beads and selection strings
    head_bead = 'GL1'
    tail_bead = 'GL2'
    head_sel = f'resname POPC and name {head_bead}'

    # 1. Order Parameter P2 vs time
    times, P2 = compute_order_parameter(u, head_bead, tail_bead)
    np.savetxt(os.path.join(args.outdir, 'order_P2.txt'), np.column_stack((times, P2)))
    save_line_plot(times, P2, 'Time (ps)', 'P2', 'Lipid Tail Order Parameter P2',
                   os.path.join(args.outdir, 'order_P2.png'))

    # 2. Tilt distribution histogram
    centers, hist = compute_tilt_distribution(u, head_bead, tail_bead)
    np.savetxt(os.path.join(args.outdir, 'tilt_distribution.txt'), np.column_stack((centers, hist)))
    save_histogram(centers, hist, 'Tilt Angle (degrees)', 'Probability Density', 'Tail Tilt Distribution',
                   os.path.join(args.outdir, 'tilt_distribution.png'))

    # 3. Undulation spectrum
    freqs, power = compute_undulation_spectrum(u, head_sel)
    np.savetxt(os.path.join(args.outdir, 'undulation_spectrum.txt'), np.column_stack((freqs, power)))
    save_line_plot(freqs, power, 'Frequency (1/ps)', 'Power', 'Membrane Undulation Spectrum',
                   os.path.join(args.outdir, 'undulation_spectrum.png'))

    # 4. Voronoi area distribution
    areas = compute_voronoi_areas(u, head_sel)
    np.savetxt(os.path.join(args.outdir, 'voronoi_areas.txt'), areas)
    bins = np.linspace(np.nanmin(areas), np.nanmax(areas), 50)
    hist_area, edges = np.histogram(areas[np.isfinite(areas)], bins=bins, density=True)
    centers_area = 0.5 * (edges[:-1] + edges[1:])
    save_histogram(centers_area, hist_area, 'Area (nm^2)', 'Density', 'Voronoi Area Distribution',
                   os.path.join(args.outdir, 'voronoi_area_distribution.png'))

    # 5. Void frequency vs time
    times_v, freq_v = compute_void_frequency(u, head_sel)
    np.savetxt(os.path.join(args.outdir, 'void_frequency.txt'), np.column_stack((times_v, freq_v)))
    save_line_plot(times_v, freq_v, 'Time (ps)', 'Number of Voids', 'Packing Defect Frequency',
                   os.path.join(args.outdir, 'void_frequency.png'))

    print("Analysis complete. Data and plots saved in", args.outdir)
#!/usr/bin/env python3
"""
CG Membrane Analysis Script
Performs:
 1. Lipid tail order parameter (P2)
 2. Tail tilt distribution
 3. Membrane undulation spectrum
 4. Voronoi tessellation area distribution
 5. Void (packing defect) frequency
Generates data files and English-language plots in the output directory.
Usage: membrane_analysis.py -p top.pdb -x traj.xtc -o results/
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist
from scipy.fft import fft, fftfreq

# 1. Tail order parameter P2
def compute_order_parameter(universe, head_bead, tail_bead):
    """Calculate tail order parameter P2 for each frame."""
    times = []
    p2_vals = []
    for ts in ProgressBar(universe.trajectory):
        vals = []
        for res in universe.select_atoms('resname POPC').residues:
            head = res.atoms.select_atoms(f'name {head_bead}')
            tail = res.atoms.select_atoms(f'name {tail_bead}')
            if len(head) == 1 and len(tail) == 1:
                v = tail.positions[0] - head.positions[0]
                norm = np.linalg.norm(v)
                if norm > 0:
                    cos = v[2] / norm
                    vals.append(0.5 * (3 * cos**2 - 1))
        times.append(ts.time)
        p2_vals.append(np.mean(vals) if vals else np.nan)
    return np.array(times), np.array(p2_vals)

# 2. Tilt distribution
def compute_tilt_distribution(universe, head_bead, tail_bead, bins=50):
    """Calculate tilt angle distribution for lipid tails."""
    thetas = []
    for ts in ProgressBar(universe.trajectory):
        for res in universe.select_atoms('resname POPC').residues:
            head = res.atoms.select_atoms(f'name {head_bead}')
            tail = res.atoms.select_atoms(f'name {tail_bead}')
            if len(head) == 1 and len(tail) == 1:
                v = tail.positions[0] - head.positions[0]
                norm = np.linalg.norm(v)
                if norm > 0:
                    theta = np.degrees(np.arccos(np.clip(v[2] / norm, -1.0, 1.0)))
                    thetas.append(theta)
    thetas = np.array(thetas)
    thetas = thetas[np.isfinite(thetas)]
    hist, edges = np.histogram(thetas, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

# 3. Undulation spectrum
def compute_undulation_spectrum(universe, head_sel, frame_interval=1):
    height_list = []
    for ts in universe.trajectory[::frame_interval]:
        head = universe.select_atoms(head_sel)
        height_list.append(head.positions[:, 2])
    heights = np.array(height_list)
    heights -= np.nanmean(heights, axis=0)
    fft_vals = fft(heights, axis=0)
    power = np.abs(fft_vals) ** 2
    N = heights.shape[0]
    dt = frame_interval * universe.trajectory.dt
    freqs = fftfreq(N, d=dt)
    half = slice(1, N // 2)
    return freqs[half], np.nanmean(power[half, :], axis=1)

# 4. Voronoi area distribution
def compute_voronoi_areas(universe, head_sel):
    head = universe.select_atoms(head_sel)
    pts = head.positions[:, :2]
    vor = Voronoi(pts)
    areas = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if not region or -1 in region:
            areas.append(np.nan)
            continue
        poly = np.array([vor.vertices[i] for i in region])
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)
    return np.array(areas)

# 5. Void frequency (packing defects)
def compute_void_frequency(universe, head_sel, threshold=0.5):
    times = []
    freq = []
    for ts in universe.trajectory:
        head = universe.select_atoms(head_sel)
        coords = head.positions[:, :2]
        dists = pdist(coords)
        voids = np.sum(dists > threshold)
        times.append(ts.time)
        freq.append(voids)
    return np.array(times), np.array(freq)

# Plotting utilities
def save_line_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_histogram(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.bar(x, y, width=x[1] - x[0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pdb', required=True)
    parser.add_argument('-x', '--xtc', required=True)
    parser.add_argument('-o', '--outdir', default='results')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    u = mda.Universe(args.pdb, args.xtc)

    # Define beads and selection strings
    head_bead = 'PO4'
    tail_bead = 'C4A' #'C4B'
    head_sel = f'resname POPC and name {head_bead}'

    # 1. Order Parameter P2 vs time
    times, P2 = compute_order_parameter(u, head_bead, tail_bead)
    np.savetxt(os.path.join(args.outdir, 'order_P2.txt'), np.column_stack((times, P2)))
    save_line_plot(times, P2, 'Time (ps)', 'P2', 'Lipid Tail Order Parameter P2',
                   os.path.join(args.outdir, 'order_P2.png'))

    # 2. Tilt distribution histogram
    centers, hist = compute_tilt_distribution(u, head_bead, tail_bead)
    np.savetxt(os.path.join(args.outdir, 'tilt_distribution.txt'), np.column_stack((centers, hist)))
    save_histogram(centers, hist, 'Tilt Angle (degrees)', 'Probability Density', 'Tail Tilt Distribution',
                   os.path.join(args.outdir, 'tilt_distribution.png'))

    # 3. Undulation spectrum
    freqs, power = compute_undulation_spectrum(u, head_sel)
    np.savetxt(os.path.join(args.outdir, 'undulation_spectrum.txt'), np.column_stack((freqs, power)))
    save_line_plot(freqs, power, 'Frequency (1/ps)', 'Power', 'Membrane Undulation Spectrum',
                   os.path.join(args.outdir, 'undulation_spectrum.png'))

    # 4. Voronoi area distribution
    areas = compute_voronoi_areas(u, head_sel)
    np.savetxt(os.path.join(args.outdir, 'voronoi_areas.txt'), areas)
    bins = np.linspace(np.nanmin(areas), np.nanmax(areas), 50)
    hist_area, edges = np.histogram(areas[np.isfinite(areas)], bins=bins, density=True)
    centers_area = 0.5 * (edges[:-1] + edges[1:])
    save_histogram(centers_area, hist_area, 'Area (nm^2)', 'Density', 'Voronoi Area Distribution',
                   os.path.join(args.outdir, 'voronoi_area_distribution.png'))

    # 5. Void frequency vs time
    times_v, freq_v = compute_void_frequency(u, head_sel)
    np.savetxt(os.path.join(args.outdir, 'void_frequency.txt'), np.column_stack((times_v, freq_v)))
    save_line_plot(times_v, freq_v, 'Time (ps)', 'Number of Voids', 'Packing Defect Frequency',
                   os.path.join(args.outdir, 'void_frequency.png'))

    print("Analysis complete. Data and plots saved in", args.outdir)
