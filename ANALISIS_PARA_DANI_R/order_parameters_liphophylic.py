import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import lipyphilic as lpp
except ImportError:
    print("Error: lipyphilic not installed. Install with: pip install lipyphilic")
    exit()

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Load universe and fill missing elements
universe = mda.Universe("martini_agua.pdb", "martini_agua.xtc")
universe.guess_TopologyAttrs(context='default', to_guess=['elements'])

# 2. Analysis parameters
start_frame = 0
stop_frame = None
step_frame = 1

def calculate_scc_statistics(scc_data):
    """Calculate statistics from SCC data"""
    stats = {
        'avg_scc': np.mean(scc_data, axis=1),  # Average over lipids for each frame
        'std_scc': np.std(scc_data, axis=1),   # Std over lipids for each frame
        'min_scc': np.min(scc_data),
        'max_scc': np.max(scc_data)
    }
    return stats

def analyze_scc_by_position(scc_data, n_positions):
    """Analyze SCC data by bead position"""
    # Assuming scc_data has shape (n_frames, n_lipids, n_positions)
    # or needs to be reshaped appropriately
    
    if scc_data.ndim == 2:
        # If data is (n_frames, n_lipids*n_positions), reshape it
        n_frames, total_values = scc_data.shape
        n_lipids = total_values // n_positions
        scc_data = scc_data.reshape(n_frames, n_lipids, n_positions)
    
    # Calculate statistics by position
    pos_means = np.mean(scc_data, axis=(0, 1))  # Mean over frames and lipids
    pos_stds = np.std(scc_data, axis=(0, 1))    # Std over frames and lipids
    
    # Time evolution for each position
    time_evolution = np.mean(scc_data, axis=1)  # Mean over lipids for each frame
    time_stds = np.std(scc_data, axis=1)        # Std over lipids for each frame
    
    return pos_means, pos_stds, time_evolution, time_stds

print("\n=== CALCULATING SCC USING LIPYPHILIC ===")

# 3. Run SCC analysis for both chains
results = {}

# Analysis for sn1 chain (??A atoms)
print("Analyzing sn1 chain (??A atoms)...")
try:
    scc_sn1 = lpp.analysis.SCC(
        universe=universe,
        tail_sel="name ??A"  # sn1 chain atoms
    )
    scc_sn1.run(start=start_frame, stop=stop_frame, step=step_frame)
    
    # Get SCC data
    scc_data_sn1 = None
    for attr_name in ['scc', 'SCC', 'results', 'values']:
        if hasattr(scc_sn1, attr_name):
            scc_data_sn1 = getattr(scc_sn1, attr_name)
            print(f"Found sn1 SCC data in attribute: {attr_name}")
            break
    
    if scc_data_sn1 is None:
        print("Available sn1 SCC attributes:", [attr for attr in dir(scc_sn1) if not attr.startswith('_')])
        raise AttributeError("Cannot find sn1 SCC data attribute")
    
    print(f"sn1 SCC data shape: {scc_data_sn1.shape}")
    scc_stats_sn1 = calculate_scc_statistics(scc_data_sn1)
    results['scc_sn1'] = scc_stats_sn1
    
except Exception as e:
    print(f"Error calculating sn1 SCC: {e}")
    scc_data_sn1 = None

# Analysis for sn2 chain (??B atoms)
print("Analyzing sn2 chain (??B atoms)...")
try:
    scc_sn2 = lpp.analysis.SCC(
        universe=universe,
        tail_sel="name ??B"  # sn2 chain atoms
    )
    scc_sn2.run(start=start_frame, stop=stop_frame, step=step_frame)
    
    # Get SCC data
    scc_data_sn2 = None
    for attr_name in ['scc', 'SCC', 'results', 'values']:
        if hasattr(scc_sn2, attr_name):
            scc_data_sn2 = getattr(scc_sn2, attr_name)
            print(f"Found sn2 SCC data in attribute: {attr_name}")
            break
    
    if scc_data_sn2 is None:
        print("Available sn2 SCC attributes:", [attr for attr in dir(scc_sn2) if not attr.startswith('_')])
        raise AttributeError("Cannot find sn2 SCC data attribute")
    
    print(f"sn2 SCC data shape: {scc_data_sn2.shape}")
    scc_stats_sn2 = calculate_scc_statistics(scc_data_sn2)
    results['scc_sn2'] = scc_stats_sn2
    
except Exception as e:
    print(f"Error calculating sn2 SCC: {e}")
    scc_data_sn2 = None

# Combined analysis
print("Analyzing combined chains (??A or ??B atoms)...")
try:
    scc_combined = lpp.analysis.SCC(
        universe=universe,
        tail_sel="name ??A or name ??B"  # Both chains
    )
    scc_combined.run(start=start_frame, stop=stop_frame, step=step_frame)
    
    # Get SCC data
    scc_data_combined = None
    for attr_name in ['scc', 'SCC', 'results', 'values']:
        if hasattr(scc_combined, attr_name):
            scc_data_combined = getattr(scc_combined, attr_name)
            print(f"Found combined SCC data in attribute: {attr_name}")
            break
    
    if scc_data_combined is None:
        print("Available combined SCC attributes:", [attr for attr in dir(scc_combined) if not attr.startswith('_')])
        raise AttributeError("Cannot find combined SCC data attribute")
    
    print(f"Combined SCC data shape: {scc_data_combined.shape}")
    scc_stats_combined = calculate_scc_statistics(scc_data_combined)
    results['scc_combined'] = scc_stats_combined
    
except Exception as e:
    print(f"Error calculating combined SCC: {e}")
    scc_data_combined = None

# 4. Get time information
frame_times = []
for ts in universe.trajectory[start_frame:stop_frame:step_frame]:
    frame_times.append(ts.time)
frame_times = np.array(frame_times)

# 5. Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Time evolution comparison between chains
if scc_data_sn1 is not None and scc_data_sn2 is not None:
    scc1_means = results['scc_sn1']['avg_scc']
    scc1_stds = results['scc_sn1']['std_scc']
    scc2_means = results['scc_sn2']['avg_scc']
    scc2_stds = results['scc_sn2']['std_scc']
    
    # Ensure time and data arrays have same length
    min_len = min(len(frame_times), len(scc1_means), len(scc2_means))
    time_plot = frame_times[:min_len]
    
    ax1.plot(time_plot, scc1_means[:min_len], color='#2E86AB', linewidth=2, label='sn1 chain', alpha=0.8)
    ax1.fill_between(time_plot, 
                     scc1_means[:min_len] - scc1_stds[:min_len], 
                     scc1_means[:min_len] + scc1_stds[:min_len], 
                     alpha=0.3, color='#2E86AB')
    
    ax1.plot(time_plot, scc2_means[:min_len], color='#A23B72', linewidth=2, label='sn2 chain', alpha=0.8)
    ax1.fill_between(time_plot, 
                     scc2_means[:min_len] - scc2_stds[:min_len], 
                     scc2_means[:min_len] + scc2_stds[:min_len], 
                     alpha=0.3, color='#A23B72')

ax1.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Order Parameter $\\langle S_{CC} \\rangle$', fontsize=12, fontweight='bold')
ax1.set_title('SCC Time Evolution by Chain', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper right', ncol=2)

# Plot 2: Combined time evolution
if scc_data_combined is not None:
    combined_means = results['scc_combined']['avg_scc']
    combined_stds = results['scc_combined']['std_scc']
    
    min_len = min(len(frame_times), len(combined_means))
    time_plot = frame_times[:min_len]
    
    ax2.plot(time_plot, combined_means[:min_len], color='#F18F01', linewidth=3, alpha=0.8)
    ax2.fill_between(time_plot, 
                     combined_means[:min_len] - combined_stds[:min_len], 
                     combined_means[:min_len] + combined_stds[:min_len], 
                     alpha=0.3, color='#F18F01')

ax2.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Order Parameter $\\langle S_{CC} \\rangle$', fontsize=12, fontweight='bold')
ax2.set_title('Combined SCC Time Evolution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Average order parameter comparison
if scc_data_sn1 is not None and scc_data_sn2 is not None:
    chains = ['sn1 chain', 'sn2 chain']
    means = [np.mean(scc1_means), np.mean(scc2_means)]
    stds = [np.mean(scc1_stds), np.mean(scc2_stds)]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax3.bar(chains, means, yerr=stds, capsize=8, alpha=0.7, color=colors)
    ax3.set_ylabel('Average Order Parameter $\\langle S_{CC} \\rangle$', fontsize=12, fontweight='bold')
    ax3.set_title('Average SCC Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Difference between chains over time
if scc_data_sn1 is not None and scc_data_sn2 is not None:
    diff_means = scc1_means[:min_len] - scc2_means[:min_len]
    diff_stds = np.sqrt(scc1_stds[:min_len]**2 + scc2_stds[:min_len]**2)
    
    ax4.plot(time_plot, diff_means, color='#C73E1D', linewidth=2, alpha=0.8)
    ax4.fill_between(time_plot, diff_means - diff_stds, diff_means + diff_stds, 
                     alpha=0.3, color='#C73E1D')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

ax4.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
ax4.set_ylabel('$\\langle S_{CC} \\rangle_{sn1} - \\langle S_{CC} \\rangle_{sn2}$', fontsize=12, fontweight='bold')
ax4.set_title('Chain Difference Over Time', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Overall styling
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

plt.tight_layout(pad=3.0)
plt.savefig("lipyphilic_scc_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# 6. Print summary statistics
print(f"\n{'='*70}")
print(f"LiPyphilic SCC Analysis Summary")
print(f"{'='*70}")
print(f"Simulation time: {frame_times[0]:.1f} - {frame_times[-1]:.1f} ps")
print(f"Number of frames: {len(frame_times)}")

if 'scc_sn1' in results:
    sn1_stats = results['scc_sn1']
    print(f"\nsn1 Chain Analysis:")
    print(f"  Average SCC: {np.mean(sn1_stats['avg_scc']):.4f} ± {np.mean(sn1_stats['std_scc']):.4f}")
    print(f"  SCC range: {sn1_stats['min_scc']:.4f} - {sn1_stats['max_scc']:.4f}")

if 'scc_sn2' in results:
    sn2_stats = results['scc_sn2']
    print(f"\nsn2 Chain Analysis:")
    print(f"  Average SCC: {np.mean(sn2_stats['avg_scc']):.4f} ± {np.mean(sn2_stats['std_scc']):.4f}")
    print(f"  SCC range: {sn2_stats['min_scc']:.4f} - {sn2_stats['max_scc']:.4f}")

if 'scc_combined' in results:
    combined_stats = results['scc_combined']
    print(f"\nCombined Analysis:")
    print(f"  Average SCC: {np.mean(combined_stats['avg_scc']):.4f} ± {np.mean(combined_stats['std_scc']):.4f}")
    print(f"  SCC range: {combined_stats['min_scc']:.4f} - {combined_stats['max_scc']:.4f}")

if 'scc_sn1' in results and 'scc_sn2' in results:
    mean_diff = np.mean(results['scc_sn1']['avg_scc']) - np.mean(results['scc_sn2']['avg_scc'])
    print(f"\nChain Comparison:")
    print(f"  Mean difference (sn1 - sn2): {mean_diff:.4f}")

print(f"{'='*70}")

# 7. Save data files
if 'scc_combined' in results:
    # Save combined SCC data
    combined_output = np.column_stack((frame_times[:len(results['scc_combined']['avg_scc'])],
                                     results['scc_combined']['avg_scc'],
                                     results['scc_combined']['std_scc']))
    np.savetxt('lipyphilic_scc_combined.txt', combined_output,
               header='Time (ps)\tAverage SCC\tStd SCC',
               fmt='%.6f\t%.6f\t%.6f')
    print(f"\nSaved combined SCC data to: lipyphilic_scc_combined.txt")

if 'scc_sn1' in results and 'scc_sn2' in results:
    # Save individual chain data
    sn1_output = np.column_stack((frame_times[:len(results['scc_sn1']['avg_scc'])],
                                results['scc_sn1']['avg_scc'],
                                results['scc_sn1']['std_scc']))
    np.savetxt('lipyphilic_scc_sn1.txt', sn1_output,
               header='Time (ps)\tAverage SCC sn1\tStd SCC sn1',
               fmt='%.6f\t%.6f\t%.6f')
    
    sn2_output = np.column_stack((frame_times[:len(results['scc_sn2']['avg_scc'])],
                                results['scc_sn2']['avg_scc'],
                                results['scc_sn2']['std_scc']))
    np.savetxt('lipyphilic_scc_sn2.txt', sn2_output,
               header='Time (ps)\tAverage SCC sn2\tStd SCC sn2',
               fmt='%.6f\t%.6f\t%.6f')
    
    print(f"Saved individual chain data to: lipyphilic_scc_sn1.txt and lipyphilic_scc_sn2.txt")
