import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Load universe and fill missing elements
u = mda.Universe("martini_agua.pdb", "martini_agua.xtc")
u.guess_TopologyAttrs(context='default', to_guess=['elements'])

# 2. Select POPC residues
lipids = u.select_atoms("resname POPC").residues

# 3. Define bead sequences for each chain
chain1 = ["C1A", "D2A", "C3A", "C4A"]  # sn1 chain
chain2 = ["C1B", "C2B", "C3B", "C4B"]  # sn2 chain

# Membrane normal (z-axis)
nz = np.array([0, 0, 1.0])

def calc_S(v):
    """Calculate order parameter S = 0.5*(3*cos²θ - 1)"""
    cosθ = np.dot(v, nz) / np.linalg.norm(v)
    return 0.5 * (3*cosθ**2 - 1)

# 4. Prepare data structures
n_positions = len(chain1) - 1  # number of bonds per chain
n_frames = len(u.trajectory)
n_lipids = len(lipids)

# Store S values for each chain, position, frame, and lipid
S_data_chain1 = []  # [position][frame][lipid_values]
S_data_chain2 = []

for i in range(n_positions):
    S_data_chain1.append([])
    S_data_chain2.append([])

# 5. Loop through frames
frame_times = []
for frame_idx, ts in enumerate(u.trajectory):
    frame_times.append(ts.time)
    
    # Initialize frame data for each position
    for i in range(n_positions):
        S_data_chain1[i].append([])
        S_data_chain2[i].append([])
    
    # Process each lipid
    for res in lipids:
        # Process sn1 chain (chain1)
        for i in range(n_positions):
            name_i = chain1[i]
            name_i1 = chain1[i+1]
            try:
                p1 = res.atoms.select_atoms(f"name {name_i}")[0].position
                p2 = res.atoms.select_atoms(f"name {name_i1}")[0].position
                S = calc_S(p2 - p1)
                S_data_chain1[i][frame_idx].append(S)
            except IndexError:
                continue
        
        # Process sn2 chain (chain2)
        for i in range(n_positions):
            name_i = chain2[i]
            name_i1 = chain2[i+1]
            try:
                p1 = res.atoms.select_atoms(f"name {name_i}")[0].position
                p2 = res.atoms.select_atoms(f"name {name_i1}")[0].position
                S = calc_S(p2 - p1)
                S_data_chain2[i][frame_idx].append(S)
            except IndexError:
                continue

# 6. Calculate statistics
def calculate_stats(data):
    """Calculate mean and std for each position across all frames and lipids"""
    means = []
    stds = []
    
    for pos_data in data:
        all_values = []
        for frame_data in pos_data:
            all_values.extend(frame_data)
        
        if all_values:
            means.append(np.mean(all_values))
            stds.append(np.std(all_values))
        else:
            means.append(0)
            stds.append(0)
    
    return np.array(means), np.array(stds)

# Calculate statistics for both chains
S_mean_chain1, S_std_chain1 = calculate_stats(S_data_chain1)
S_mean_chain2, S_std_chain2 = calculate_stats(S_data_chain2)

# 7. Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Order parameter by position for both chains
positions = np.arange(1, n_positions + 1)

# Chain 1 (sn1)
ax1.errorbar(positions, S_mean_chain1, yerr=S_std_chain1, 
             marker='o', linewidth=2.5, markersize=8, capsize=5,
             color='#2E86AB', label='sn1 chain', alpha=0.8)

# Chain 2 (sn2)
ax1.errorbar(positions, S_mean_chain2, yerr=S_std_chain2, 
             marker='s', linewidth=2.5, markersize=8, capsize=5,
             color='#A23B72', label='sn2 chain', alpha=0.8)

ax1.set_xticks(positions)
ax1.set_xlabel('Bead Position (i → i+1)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Order Parameter $\\langle S \\rangle$', fontsize=12, fontweight='bold')
ax1.set_title('Order Parameter by Bead Position', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Combined average
S_mean_combined = (S_mean_chain1 + S_mean_chain2) / 2
S_std_combined = np.sqrt((S_std_chain1**2 + S_std_chain2**2) / 2)

ax2.errorbar(positions, S_mean_combined, yerr=S_std_combined, 
             marker='D', linewidth=3, markersize=8, capsize=5,
             color='#F18F01', label='Combined chains', alpha=0.8)

ax2.set_xticks(positions)
ax2.set_xlabel('Bead Position (i → i+1)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Order Parameter $\\langle S \\rangle$', fontsize=12, fontweight='bold')
ax2.set_title('Combined Order Parameter', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 3: Time evolution for middle position (example)
middle_pos = n_positions // 2
time_means_chain1 = []
time_stds_chain1 = []
time_means_chain2 = []
time_stds_chain2 = []

for frame_idx in range(len(frame_times)):
    if S_data_chain1[middle_pos][frame_idx]:
        time_means_chain1.append(np.mean(S_data_chain1[middle_pos][frame_idx]))
        time_stds_chain1.append(np.std(S_data_chain1[middle_pos][frame_idx]))
    else:
        time_means_chain1.append(0)
        time_stds_chain1.append(0)
    
    if S_data_chain2[middle_pos][frame_idx]:
        time_means_chain2.append(np.mean(S_data_chain2[middle_pos][frame_idx]))
        time_stds_chain2.append(np.std(S_data_chain2[middle_pos][frame_idx]))
    else:
        time_means_chain2.append(0)
        time_stds_chain2.append(0)

time_means_chain1 = np.array(time_means_chain1)
time_stds_chain1 = np.array(time_stds_chain1)
time_means_chain2 = np.array(time_means_chain2)
time_stds_chain2 = np.array(time_stds_chain2)

# Plot time evolution with uncertainty bands
ax3.plot(frame_times, time_means_chain1, color='#2E86AB', linewidth=2, label='sn1 chain')
ax3.fill_between(frame_times, time_means_chain1 - time_stds_chain1, 
                 time_means_chain1 + time_stds_chain1, alpha=0.3, color='#2E86AB')

ax3.plot(frame_times, time_means_chain2, color='#A23B72', linewidth=2, label='sn2 chain')
ax3.fill_between(frame_times, time_means_chain2 - time_stds_chain2, 
                 time_means_chain2 + time_stds_chain2, alpha=0.3, color='#A23B72')

ax3.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Order Parameter $\\langle S \\rangle$', fontsize=12, fontweight='bold')
ax3.set_title(f'Time Evolution Mean Value', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10,loc='upper right',ncol=2)

# Plot 4: Difference between chains
S_diff = S_mean_chain1 - S_mean_chain2
S_diff_err = np.sqrt(S_std_chain1**2 + S_std_chain2**2)

ax4.errorbar(positions, S_diff, yerr=S_diff_err, 
             marker='v', linewidth=2.5, markersize=8, capsize=5,
             color='#C73E1D', alpha=0.8)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

ax4.set_xticks(positions)
ax4.set_xlabel('Bead Position (i → i+1)', fontsize=12, fontweight='bold')
ax4.set_ylabel('$\\langle S \\rangle_{sn1} - \\langle S \\rangle_{sn2}$', fontsize=12, fontweight='bold')
ax4.set_title('Chain Difference (sn1 - sn2)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Overall styling
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

plt.tight_layout(pad=3.0)
plt.savefig("order_parameter_comprehensive.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. Print summary statistics
print(f"\n{'='*70}")
print(f"POPC Order Parameter Analysis Summary")
print(f"{'='*70}")
print(f"Simulation time: {frame_times[0]:.1f} - {frame_times[-1]:.1f} ps")
print(f"Number of frames: {len(frame_times)}")
print(f"Number of lipids: {len(lipids)}")
print(f"Bead positions analyzed: {n_positions}")

print(f"\nsn1 Chain ({' -> '.join(chain1)}):")
for i, (mean, std) in enumerate(zip(S_mean_chain1, S_std_chain1)):
    bond = f"{chain1[i]} -> {chain1[i+1]}"
    print(f"  Position {i+1:2d} ({bond:12s}): {mean:6.3f} ± {std:5.3f}")

print(f"\nsn2 Chain ({' -> '.join(chain2)}):")
for i, (mean, std) in enumerate(zip(S_mean_chain2, S_std_chain2)):
    bond = f"{chain2[i]} -> {chain2[i+1]}"
    print(f"  Position {i+1:2d} ({bond:12s}): {mean:6.3f} ± {std:5.3f}")

print(f"\nCombined Analysis:")
print(f"  Overall mean sn1: {np.mean(S_mean_chain1):.3f}")
print(f"  Overall mean sn2: {np.mean(S_mean_chain2):.3f}")
print(f"  Mean difference:   {np.mean(S_diff):.3f}")
print(f"{'='*70}")
