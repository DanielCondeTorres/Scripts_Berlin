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

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist

def compute_voronoi_areas(universe, head_sel, boundary=None):
    """
    Compute Voronoi areas for head groups
    
    Parameters:
    -----------
    universe : MDAnalysis Universe object
        The molecular dynamics universe
    head_sel : str
        Selection string for head groups
    boundary : tuple, optional
        (xmin, xmax, ymin, ymax) boundary box for clipping Voronoi regions
        If None, will use the system box dimensions
        
    Returns:
    --------
    np.array
        Array of Voronoi areas for each point
    """
    head = universe.select_atoms(head_sel)
    pts = head.positions[:, :2]
    
    # Set boundary based on system box if not provided
    if boundary is None:
        box = universe.dimensions[:2]
        boundary = (0, box[0], 0, box[1])
    
    # Create Voronoi diagram
    vor = Voronoi(pts)
    
    # Calculate areas
    areas = np.zeros(len(pts))
    
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        # Skip if region is empty or unbounded
        if not region or -1 in region:
            # For unbounded regions, we can approximate by creating a bounded region
            # using the boundary of the simulation box
            vertices = _get_bounded_region(vor, i, boundary)
            if vertices is None or len(vertices) < 3:
                areas[i] = np.nan
                continue
        else:
            vertices = np.array([vor.vertices[j] for j in region])
            
            # Clip polygon to boundary if needed
            if boundary is not None:
                vertices = _clip_polygon_to_box(vertices, boundary)
                if len(vertices) < 3:
                    areas[i] = np.nan
                    continue
        
        # Calculate polygon area using shoelace formula
        x, y = vertices[:, 0], vertices[:, 1]
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 
                            x[-1] * y[0] - x[0] * y[-1])
        areas[i] = area
    
    return areas

def _clip_polygon_to_box(vertices, boundary):
    """
    Clip a polygon to a rectangular boundary
    
    Parameters:
    -----------
    vertices : np.array
        Nx2 array of polygon vertices
    boundary : tuple
        (xmin, xmax, ymin, ymax) boundary box
    
    Returns:
    --------
    np.array
        Clipped polygon vertices
    """
    xmin, xmax, ymin, ymax = boundary
    
    # Simple clipping - discard points outside the box and add intersection points
    # This is a simplified approach and doesn't handle all cases correctly
    # For a complete implementation, consider using a library like Shapely
    
    # For now, just return vertices inside the box
    vertices = np.array([v for v in vertices if 
                       (xmin <= v[0] <= xmax and ymin <= v[1] <= ymax)])
    
    return vertices

def _get_bounded_region(vor, point_idx, boundary):
    """
    Get a bounded region for an unbounded Voronoi cell
    
    Parameters:
    -----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram
    point_idx : int
        Index of the point
    boundary : tuple
        (xmin, xmax, ymin, ymax) boundary box
    
    Returns:
    --------
    np.array or None
        Bounded polygon vertices or None if cannot be calculated
    """
    xmin, xmax, ymin, ymax = boundary
    
    # This is a simplified approach - for a complete implementation, 
    # consider using a library like Shapely
    
    # Using ridge points and ridge vertices to reconstruct the region
    bounded_vertices = []
    
    # Find all ridges for this point
    for i, (p1, p2) in enumerate(vor.ridge_points):
        if p1 == point_idx or p2 == point_idx:
            v_idx = vor.ridge_vertices[i]
            
            # Skip if ridge is unbounded
            if -1 in v_idx:
                continue
                
            # Add vertices to the list
            for idx in v_idx:
                vertex = vor.vertices[idx]
                if (xmin <= vertex[0] <= xmax and ymin <= vertex[1] <= ymax):
                    bounded_vertices.append(vertex)
    
    if len(bounded_vertices) < 3:
        return None
        
    return np.array(bounded_vertices)

def compute_void_frequency(universe, head_sel, threshold=0.5):
    """
    Compute void frequency (packing defects)
    
    Parameters:
    -----------
    universe : MDAnalysis Universe object
        The molecular dynamics universe
    head_sel : str
        Selection string for head groups
    threshold : float, optional
        Distance threshold for considering a void, default 0.5
        
    Returns:
    --------
    tuple of np.array
        (times, frequencies)
    """
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
    tail_bead = 'C3A'
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
