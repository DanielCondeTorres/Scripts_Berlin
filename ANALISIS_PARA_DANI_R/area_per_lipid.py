import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze lipid bilayer properties')
    parser.add_argument('-pdb', '--pdb', required=True, help='Structure file (GRO/PDB)')
    parser.add_argument('-xtc', '--xtc', required=True, help='Trajectory file (XTC/DCD)')
    return parser.parse_args()

def calculate_area_per_lipid(universe, lipid_selection='resname POPC'):
    """Calculate area per lipid"""
    # Select lipids
    lipids = universe.select_atoms(lipid_selection)
    
    # Find leaflets - specifying cutoff and pbc options to ensure proper leaflet detection 
    L = LeafletFinder(universe, 'name P*', cutoff=15.0, pbc=True)
    
    # Check if leaflets were found
    if len(L.components) == 0:
        print("Warning: No leaflets were identified. Check your structure or selection criteria.")
        return None
    
    # Use the first leaflet
    lipid_leaflet = L.groups(0)
    
    print(f"Found {len(L.components)} leaflets")
    print(f"Leaflet 0 contains {len(lipid_leaflet)} lipids")
    
    # Calculate area per lipid for each frame
    areas = []
    for ts in universe.trajectory:
        x_positions = lipid_leaflet.positions[:, 0]
        y_positions = lipid_leaflet.positions[:, 1]
        x_max, y_max = np.max(x_positions), np.max(y_positions)
        x_min, y_min = np.min(x_positions), np.min(y_positions)
        area = (x_max - x_min) * (y_max - y_min)
        areas.append(area / len(lipid_leaflet))
    
    areas = np.array(areas)
    np.savetxt('area_per_lipid.txt', areas)
    
    print(f'Average area per lipid: {np.mean(areas):.2f}±{np.std(areas):.2f} Å²')
    return areas

def calculate_rdf(universe, selection1='name P or name PO4 or name P8', selection2='name P or name PO4 or name P8'):
    """Calculate radial distribution function"""
    # Select atoms
    group1 = universe.select_atoms(selection1)
    group2 = universe.select_atoms(selection2)
    
    # Ensure there are sufficient atoms
    print(f"Number of atoms in first selection: {len(group1)}")
    print(f"Number of atoms in second selection: {len(group2)}")
    
    if len(group1) == 0 or len(group2) == 0:
        print("Error: One or both atom selections are empty")
        return None
    
    # RDF parameters
    rdf_range = (1.0, 20.0)
    nbins = 200
    
    # Calculate RDF
    rdf = InterRDF(group1, group2, nbins=nbins, range=rdf_range, norm='density')
    rdf.run()
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(rdf.bins, rdf.rdf, label=f'RDF {selection1}-{selection2}')
    plt.xlabel('Distance (Å)')
    plt.ylabel('RDF')
    plt.title(f'Radial Distribution Function between {selection1} and {selection2}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'rdf_{selection1.replace(" ", "_")}_{selection2.replace(" ", "_")}.png')
    
    return rdf

def calculate_membrane_thickness(universe):
    """Calculate membrane thickness"""
    # Find leaflets with more robust parameters
    L = LeafletFinder(universe, 'name P or name PO4 or name P8', cutoff=15.0, pbc=True)
    
    # Check if enough leaflets were found
    if len(L.components) < 2:
        print(f"Warning: Only {len(L.components)} leaflets were identified. Need 2 for thickness calculation.")
        return None
    
    # Get the two leaflets
    leaflet1 = L.groups(0)
    leaflet2 = L.groups(1)
    
    print(f"Leaflet 1 contains {len(leaflet1)} lipids")
    print(f"Leaflet 2 contains {len(leaflet2)} lipids")
    
    # Calculate thickness over time
    thicknesses = []
    time = []
    for ts in universe.trajectory:
        z_positions_leaflet1 = leaflet1.positions[:, 2]
        z_positions_leaflet2 = leaflet2.positions[:, 2]
        mean_z_leaflet1 = np.mean(z_positions_leaflet1)
        mean_z_leaflet2 = np.mean(z_positions_leaflet2)
        thickness = np.abs(mean_z_leaflet1 - mean_z_leaflet2)
        thicknesses.append(thickness)
        time.append(ts.time)
    print('Time: ', time)
    thicknesses = np.array(thicknesses)
    np.savetxt('membrane_thickness.txt', thicknesses)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(time, thicknesses, label='Membrane Thickness')
    plt.xlabel('Time (ps)')
    plt.ylabel('Thickness (Å)')
    plt.title('Membrane Thickness over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('membrane_thickness.png')
    
    print(f'Average membrane thickness: {np.mean(thicknesses):.2f}±{np.std(thicknesses):.2f} Å')
    return thicknesses

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load structure and trajectory
    print(f"Loading universe from {args.pdb} and {args.xtc}")
    u = mda.Universe(args.pdb, args.xtc)
    print(f"System contains {len(u.atoms)} atoms")
    
    # Calculate membrane properties with error handling
    try:
        print("\n=== Calculating Area per Lipid ===")
        areas = calculate_area_per_lipid(u)
    except Exception as e:
        print(f"Error calculating area per lipid: {e}")
    
    try:
        print("\n=== Calculating RDF for Phosphate Heads ===")
        rdf = calculate_rdf(u, 'name P or name PO4', 'name P or name PO4')
    except Exception as e:
        print(f"Error calculating RDF: {e}")
    
    try:
        print("\n=== Calculating Membrane Thickness ===")
        thicknesses = calculate_membrane_thickness(u)
    except Exception as e:
        print(f"Error calculating membrane thickness: {e}")
    
    plt.show()  # Show all plots at the end

if __name__ == "__main__":
    main()
