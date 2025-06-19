#!/usr/bin/env python3
"""
Simple script for lipid area analysis compatible with different lipyphilic versions
Usage: python simple_area_analysis.py -pdb structure.pdb -xtc trajectory.xtc
"""

import argparse
import sys
import os
import MDAnalysis as mda
import lipyphilic as lpp
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Lipid area analysis')
    parser.add_argument('-pdb', required=True, help='PDB file')
    parser.add_argument('-xtc', required=True, help='XTC file')
    parser.add_argument('-sel', default="name PO4 ROH", help='Lipid selection')
    parser.add_argument('-start', type=int, default=None, help='Starting frame')
    parser.add_argument('-stop', type=int, default=None, help='Final frame')
    parser.add_argument('-step', type=int, default=None, help='Step between frames')
    parser.add_argument('-o', default="area_analysis", help='Output prefix')
    return parser.parse_args()

def check_lipyphilic_version():
    """Detects which function to use for leaflets"""
    try:
        # Try different ways to access functions
        if hasattr(lpp.analysis, 'AssignLeaflets'):
            return 'AssignLeaflets', lpp.analysis.AssignLeaflets
        elif hasattr(lpp, 'AssignLeaflets'):
            return 'AssignLeaflets', lpp.AssignLeaflets
        elif hasattr(lpp.analysis, 'LeafletFinder'):
            return 'LeafletFinder', lpp.analysis.LeafletFinder
        elif hasattr(lpp, 'LeafletFinder'):
            return 'LeafletFinder', lpp.LeafletFinder
        else:
            print("Cannot find function for leaflets")
            return None, None
    except Exception as e:
        print(f"Error detecting version: {e}")
        return None, None

def assign_leaflets(universe, lipid_sel):
    """Assigns leaflets using available function"""
    
    func_name, func = check_lipyphilic_version()
    
    if func is None:
        print("Error: Cannot find function to assign leaflets")
        sys.exit(1)
    
    print(f"Using {func_name} to assign leaflets")
    
    try:
        leaflets = func(universe=universe, lipid_sel=lipid_sel)
        leaflets.run()
        return leaflets
    except Exception as e:
        print(f"Error assigning leaflets: {e}")
        sys.exit(1)

def calculate_area_simple_method(universe, lipid_sel, start=None, stop=None, step=None):
    """Simple alternative method if there are problems with leaflets"""
    
    print("Trying alternative method without leaflets...")
    
    # Get lipids
    lipids = universe.select_atoms(lipid_sel)
    n_lipids = len(lipids)
    
    if n_lipids == 0:
        print(f"Error: No lipids found with selection: {lipid_sel}")
        return None
    
    # Determine frames to analyze
    frames = list(range(len(universe.trajectory)))
    if start is not None:
        frames = [f for f in frames if f >= start]
    if stop is not None:
        frames = [f for f in frames if f < stop]
    if step is not None:
        frames = frames[::step]
    
    areas = []
    
    for frame_idx in frames:
        universe.trajectory[frame_idx]
        
        # Calculate box area divided by number of lipids
        # This is a simple approximation
        box_area = universe.dimensions[0] * universe.dimensions[1]  # XY area
        area_per_lipid = box_area / n_lipids
        
        areas.append(area_per_lipid)
    
    return np.array(areas)

def create_time_array(n_frames, dt_ns, start_frame=0, step=1):
    """Create time array in microseconds"""
    # Convert nanoseconds to microseconds (divide by 1000)
    dt_us = dt_ns / 100000.0
    print('dt ns: ',dt_ns)
    
    # Create time points
    frame_indices = np.arange(start_frame, start_frame + n_frames * step, step)
    time_us = frame_indices * dt_us
    
    return time_us

def main():
    args = parse_args()
    
    # Validate files
    if not os.path.exists(args.pdb):
        print(f"Error: {args.pdb} does not exist")
        sys.exit(1)
    if not os.path.exists(args.xtc):
        print(f"Error: {args.xtc} does not exist")
        sys.exit(1)
    
    print(f"Loading: {args.pdb} and {args.xtc}")
    
    # Load universe
    try:
        universe = mda.Universe(args.pdb, args.xtc)
        print(f"Universe loaded: {len(universe.atoms)} atoms, {len(universe.trajectory)} frames")
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    
    # Verify selection
    try:
        test_sel = universe.select_atoms(args.sel)
        print(f"Selection '{args.sel}': {len(test_sel)} atoms found")
        if len(test_sel) == 0:
            print("Error: Selection found no atoms")
            sys.exit(1)
    except Exception as e:
        print(f"Selection error: {e}")
        sys.exit(1)
    
    # Method 1: Using complete lipyphilic
    try:
        print("\n=== METHOD 1: Using lipyphilic with leaflets ===")
        
        # Assign leaflets
        leaflets = assign_leaflets(universe, args.sel)
        
        # Calculate areas
        areas_analysis = lpp.analysis.AreaPerLipid(
            universe=universe,
            lipid_sel=args.sel,
            leaflets=leaflets.leaflets
        )
        
        areas_analysis.run(start=args.start, stop=args.stop, step=args.step)
        areas = areas_analysis.areas
        
        print(f"Data calculated: {areas.shape}")
        print(f"Average area: {np.mean(areas):.2f} ± {np.std(areas):.2f} Å²")
        
        # Save complete data
        np.savetxt(f'{args.o}_detailed.txt', areas, 
                   header='Area per lipid (Å²) - Rows: frames, Columns: lipids')
        
        # Statistics per frame
        area_per_frame = np.mean(areas, axis=1)
        
    except Exception as e:
        print(f"Error with method 1: {e}")
        print("\n=== METHOD 2: Simple approximation ===")
        
        # Simple alternative method
        area_per_frame = calculate_area_simple_method(
            universe, args.sel, args.start, args.stop, args.step
        )
        
        if area_per_frame is None:
            print("Error: Could not calculate area")
            sys.exit(1)
        
        areas = area_per_frame.reshape(-1, 1)  # Similar format
        print(f"Average area (simple method): {np.mean(area_per_frame):.2f} ± {np.std(area_per_frame):.2f} Å²")
    
    # Create time array in microseconds
    start_frame = args.start if args.start is not None else 0
    step = args.step if args.step is not None else 1
    time_us = create_time_array(len(area_per_frame), universe.trajectory.dt, start_frame, step)
    
    # Save results with time
    data_with_time = np.column_stack((time_us, area_per_frame))
    np.savetxt(f'{args.o}_per_frame.txt', data_with_time, 
               header='Time (μs)\tAverage area per frame (Å²)', 
               fmt='%.6f\t%.6f')
    
    # Statistics
    stats = {
        'mean': np.mean(areas),
        'std': np.std(areas),
        'min': np.min(areas),
        'max': np.max(areas),
        'median': np.median(areas)
    }
    
    # Save summary
    with open(f'{args.o}_summary.txt', 'w') as f:
        f.write("AREA ANALYSIS SUMMARY\n")
        f.write("="*30 + "\n")
        f.write(f"Frames analyzed: {len(area_per_frame)}\n")
        f.write(f"Time step: {universe.trajectory.dt} ns ({universe.trajectory.dt/1000:.3f} μs)\n")
        f.write(f"Total simulation time: {time_us[-1]:.3f} μs\n")
        f.write(f"Selection: {args.sel}\n")
        f.write(f"Average area: {stats['mean']:.2f} ± {stats['std']:.2f} Å²\n")
        f.write(f"Range: {stats['min']:.2f} - {stats['max']:.2f} Å²\n")
        f.write(f"Median: {stats['median']:.2f} Å²\n")
    
    # Simple plot
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(time_us, area_per_frame, 'b-', alpha=0.7)
        plt.xlabel('Time (μs)')
        plt.ylabel('Average Area (Å²)')
        plt.title('Temporal Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(areas.flatten(), bins=30, alpha=0.7, color='green')
        plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
        plt.xlabel('Area (Å²)')
        plt.ylabel('Frequency')
        plt.title('Area Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        cumulative_mean = np.cumsum(area_per_frame) / np.arange(1, len(area_per_frame) + 1)
        plt.plot(time_us, cumulative_mean, 'r-', linewidth=2)
        plt.xlabel('Time (μs)')
        plt.ylabel('Cumulative Average Area (Å²)')
        plt.title('Convergence')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.boxplot([areas.flatten()], patch_artist=True)
        plt.ylabel('Area (Å²)')
        plt.title('Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{args.o}_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved: {args.o}_plots.png")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Show final results
    print(f"\n=== FINAL RESULTS ===")
    print(f"Average area: {stats['mean']:.2f} ± {stats['std']:.2f} Å²")
    print(f"Range: {stats['min']:.2f} - {stats['max']:.2f} Å²")
    print(f"Median: {stats['median']:.2f} Å²")
    print(f"Total simulation time: {time_us[-1]:.3f} μs")
    print(f"\nFiles saved:")
    print(f"- {args.o}_summary.txt")
    print(f"- {args.o}_per_frame.txt")
    if 'areas_analysis' in locals():
        print(f"- {args.o}_detailed.txt")
    print(f"- {args.o}_plots.png")

if __name__ == "__main__":
    main()
