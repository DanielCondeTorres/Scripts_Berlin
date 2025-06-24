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
universe = mda.Universe("../vacuum.pdb", "../vacuum.xtc")
universe.guess_TopologyAttrs(context='default', to_guess=['elements'])

print(f"Trajectory info:")
print(f"  Total frames: {len(universe.trajectory)}")
print(f"  Time step: {universe.trajectory.dt} ps")
print(f"  Total time: {universe.trajectory.totaltime} ps = {universe.trajectory.totaltime/1000000:.6f} µs")

# 2. Analysis parameters - IMPORTANT: Use step=1 to analyze ALL frames
start_frame = 0
stop_frame = None  # This will use all frames
step_frame = 1     # MUST be 1 to analyze every single frame

def calculate_scc_statistics(scc_data):
    """Calculate statistics from SCC data - FIXED VERSION"""
    print(f"SCC data shape in stats calculation: {scc_data.shape}")
    
    # scc_data shape should be (n_frames, n_atoms_or_bonds)
    if scc_data.ndim == 2:
        # Average over atoms/bonds for each frame (axis=1, not axis=0)
        stats = {
            'avg_scc': np.mean(scc_data, axis=1),  # Average over atoms for each frame
            'std_scc': np.std(scc_data, axis=1),   # Std over atoms for each frame  
            'min_scc': np.min(scc_data),
            'max_scc': np.max(scc_data)
        }
    else:
        # If 1D, assume it's already averaged per frame
        stats = {
            'avg_scc': scc_data,
            'std_scc': np.zeros_like(scc_data),
            'min_scc': np.min(scc_data),
            'max_scc': np.max(scc_data)
        }
    
    print(f"Calculated avg_scc shape: {stats['avg_scc'].shape}")
    print(f"Calculated std_scc shape: {stats['std_scc'].shape}")
    
    return stats

def get_time_data(universe, start_frame, stop_frame, step_frame):
    """Get time data properly formatted in microseconds"""
    frame_times = []
    for ts in universe.trajectory[start_frame:stop_frame:step_frame]:
        frame_times.append(ts.time)
    frame_times = np.array(frame_times)
    
    # Convert to microseconds (assuming original is in ps)
    frame_times_us = frame_times / 1000000.0
    
    print(f"Time range: {frame_times_us[0]:.6f} - {frame_times_us[-1]:.6f} µs")
    print(f"Number of time points: {len(frame_times_us)}")
    
    return frame_times_us

print("\n=== CALCULATING SCC USING LIPYPHILIC ===")

# 3. Get time information FIRST
time_us = get_time_data(universe, start_frame, stop_frame, step_frame)

# 4. Run SCC analysis for both chains
results = {}

# Analysis for sn1 chain (??A atoms)
print("\nAnalyzing sn1 chain (??A atoms)...")
try:
    scc_sn1 = lpp.analysis.SCC(
        universe=universe,
        tail_sel="name ??A"  # sn1 chain atoms
    )
    
    # CRITICAL: Make sure to run with step=1 to get ALL frames
    print(f"Running SCC analysis from frame {start_frame} to {stop_frame} with step {step_frame}")
    scc_sn1.run(start=start_frame, stop=stop_frame, step=step_frame)
    print('SHAPE: ',scc_sn1.SCC.shape)    
    
    # Debug: Check what's actually in the SCC object
    print("Available SCC attributes:", [attr for attr in dir(scc_sn1) if not attr.startswith('_')])
    
    # Try to get times from SCC object
    if hasattr(scc_sn1, 'times'):
        scc_times = scc_sn1.times
        print(f"SCC times shape: {scc_times.shape}")
        print(f"SCC times range: {scc_times[0]} - {scc_times[-1]} ps")
        # Update our time array to match SCC analysis
        time_us = scc_times / 1000000.0  # Convert to µs
        print(f"Updated time array to match SCC: {len(time_us)} points")
    
    # Get SCC data
    scc_data_sn1 = None
    for attr_name in ['scc', 'SCC', 'results', 'values']:
        if hasattr(scc_sn1, attr_name):
            scc_data_sn1 = getattr(scc_sn1, attr_name)
            print(f"Found sn1 SCC data in attribute: {attr_name}")
            break
    
    if scc_data_sn1 is None:
        raise AttributeError("Cannot find sn1 SCC data attribute")
    
    # Transponer para que shape = (n_frames, n_lipidos)
    print(f"Original sn1 SCC data shape: {scc_data_sn1.shape}")
    scc_data_sn1 = scc_data_sn1.T
    print(f"Transposed sn1 SCC data shape: {scc_data_sn1.shape}")
    
    # Ensure time and data arrays match
    min_len = min(len(time_us), scc_data_sn1.shape[0])
    time_us = time_us[:min_len]
    scc_data_sn1 = scc_data_sn1[:min_len]
    
    print(f"After synchronization - Time: {len(time_us)}, SCC data: {scc_data_sn1.shape}")
    
    scc_stats_sn1 = calculate_scc_statistics(scc_data_sn1)
    results['scc_sn1'] = scc_stats_sn1
    
except Exception as e:
    print(f"Error calculating sn1 SCC: {e}")
    import traceback
    traceback.print_exc()
    scc_data_sn1 = None

# Analysis for sn2 chain (??B atoms)
print("\nAnalyzing sn2 chain (??B atoms)...")
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
        raise AttributeError("Cannot find sn2 SCC data attribute")
    
    # Transponer para que shape = (n_frames, n_lipidos)
    print(f"Original sn2 SCC data shape: {scc_data_sn2.shape}")
    scc_data_sn2 = scc_data_sn2.T
    print(f"Transposed sn2 SCC data shape: {scc_data_sn2.shape}")
    
    # Ensure consistency with time array
    min_len = min(len(time_us), scc_data_sn2.shape[0])
    scc_data_sn2 = scc_data_sn2[:min_len]
    time_us = time_us[:min_len]
    
    scc_stats_sn2 = calculate_scc_statistics(scc_data_sn2)
    results['scc_sn2'] = scc_stats_sn2
    
except Exception as e:
    print(f"Error calculating sn2 SCC: {e}")
    scc_data_sn2 = None

# Combined analysis
print("\nAnalyzing combined chains (??A or ??B atoms)...")
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
        raise AttributeError("Cannot find combined SCC data attribute")
    
    # Transponer para que shape = (n_frames, n_lipidos)
    print(f"Original combined SCC data shape: {scc_data_combined.shape}")
    scc_data_combined = scc_data_combined.T
    print(f"Transposed combined SCC data shape: {scc_data_combined.shape}")
    
    # Ensure consistency with time array
    min_len = min(len(time_us), scc_data_combined.shape[0])
    scc_data_combined = scc_data_combined[:min_len]
    time_us = time_us[:min_len]
    
    scc_stats_combined = calculate_scc_statistics(scc_data_combined)
    results['scc_combined'] = scc_stats_combined
    
except Exception as e:
    print(f"Error calculating combined SCC: {e}")
    scc_data_combined = None

print(f"\n🔍 DIAGNOSIS:")
print(f"Expected frames (from trajectory): {len(universe.trajectory)}")
print(f"Time points we calculated: {len(time_us) if 'time_us' in locals() else 'N/A'}")
print(f"SCC frames we got: {scc_data_sn1.shape[0] if scc_data_sn1 is not None else 'N/A'}")

# 5. Create comprehensive plots with proper dimension handling
if len(time_us) > 0 and any(key in results for key in ['scc_sn1', 'scc_sn2', 'scc_combined']):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    print(f"\n📊 PLOTTING:")
    print(f"Final time array length: {len(time_us)}")
    print(f"Time range: {time_us[0]:.6f} - {time_us[-1]:.6f} µs")
    
    # Verify dimensions before plotting
    if 'scc_sn1' in results:
        print(f"sn1 avg_scc shape: {results['scc_sn1']['avg_scc'].shape}")
    if 'scc_sn2' in results:
        print(f"sn2 avg_scc shape: {results['scc_sn2']['avg_scc'].shape}")
    if 'scc_combined' in results:
        print(f"combined avg_scc shape: {results['scc_combined']['avg_scc'].shape}")
    
    # Plot 1: Time evolution comparison between chains
    if 'scc_sn1' in results and 'scc_sn2' in results:
        scc1_means = results['scc_sn1']['avg_scc']
        scc1_stds = results['scc_sn1']['std_scc']
        scc2_means = results['scc_sn2']['avg_scc']
        scc2_stds = results['scc_sn2']['std_scc']
        
        # Ensure arrays match time_us length
        min_len = min(len(time_us), len(scc1_means), len(scc2_means))
        time_plot = time_us[:min_len]
        scc1_means = scc1_means[:min_len]
        scc1_stds = scc1_stds[:min_len]
        scc2_means = scc2_means[:min_len]
        scc2_stds = scc2_stds[:min_len]
        
        ax1.plot(time_plot, scc1_means, color='#2E86AB', linewidth=2, label='sn1 chain', alpha=0.8)
        ax1.fill_between(time_plot, 
                         scc1_means - scc1_stds, 
                         scc1_means + scc1_stds, 
                         alpha=0.3, color='#2E86AB')
        
        ax1.plot(time_plot, scc2_means, color='#A23B72', linewidth=2, label='sn2 chain', alpha=0.8)
        ax1.fill_between(time_plot, 
                         scc2_means - scc2_stds, 
                         scc2_means + scc2_stds, 
                         alpha=0.3, color='#A23B72')
    
    ax1.set_xlabel('Time (µs)', fontsize=12, fontweight='bold');ax1.set_xlim(0, 1)
    ax1.set_xlim(0, time_us[-1])
    ax1.set_ylabel('Order Parameter $\\langle S_{CC} \\rangle$', fontsize=12, fontweight='bold')
    ax1.set_title(f'SCC Time Evolution by Chain ({len(time_us)} points)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right', ncol=2)
    
    # Plot 2: Combined time evolution
    if 'scc_combined' in results:
        combined_means = results['scc_combined']['avg_scc']
        combined_stds = results['scc_combined']['std_scc']
        
        # Ensure arrays match time_us length
        min_len = min(len(time_us), len(combined_means))
        time_plot = time_us[:min_len]
        combined_means = combined_means[:min_len]
        combined_stds = combined_stds[:min_len]
        
        ax2.plot(time_plot, combined_means, color='#F18F01', linewidth=3, alpha=0.8)
        ax2.fill_between(time_plot, 
                         combined_means - combined_stds, 
                         combined_means + combined_stds, 
                         alpha=0.3, color='#F18F01')
    
    ax2.set_xlabel('Time (µs)', fontsize=12, fontweight='bold');ax2.set_xlim(0, 1)
    ax2.set_xlim(0, time_us[-1])
    ax2.set_ylabel('Order Parameter $\\langle S_{CC} \\rangle$', fontsize=12, fontweight='bold')
    ax2.set_title(f'Combined SCC Time Evolution ({len(time_us)} points)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution histograms
    if 'scc_sn1' in results and 'scc_sn2' in results:
        ax3.hist(results['scc_sn1']['avg_scc'], bins=50, alpha=0.7, 
                color='#2E86AB', label='sn1 chain', density=True)
        ax3.hist(results['scc_sn2']['avg_scc'], bins=50, alpha=0.7, 
                color='#A23B72', label='sn2 chain', density=True)
        ax3.set_xlabel('Order Parameter $S_{CC}$', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax3.set_title('SCC Distribution by Chain', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    if len(results) > 0:
        labels = []
        means = []
        stds = []
        
        for key, data in results.items():
            labels.append(key.replace('scc_', '').replace('_', ' ').title())
            means.append(np.mean(data['avg_scc']))
            stds.append(np.std(data['avg_scc']))
        
        x_pos = np.arange(len(labels))
        bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=['#2E86AB', '#A23B72', '#F18F01'][:len(labels)], 
                      alpha=0.8)
        
        ax4.set_xlabel('Chain Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Mean Order Parameter $\\langle S_{CC} \\rangle$', fontsize=12, fontweight='bold')
        ax4.set_title('Average SCC by Chain Type', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.savefig("lipyphilic_scc_analysis_fixed.png", dpi=300, bbox_inches='tight')
    plt.show()

# 6. Print summary statistics
print(f"\n{'='*70}")
print(f"LiPyphilic SCC Analysis Summary")
print(f"{'='*70}")
print(f"Simulation time: {time_us[0]:.6f} - {time_us[-1]:.6f} µs")
print(f"Number of analysis points: {len(time_us)}")
print(f"Total trajectory frames: {len(universe.trajectory)}")
print(f"Analysis coverage: {len(time_us)/len(universe.trajectory)*100:.1f}%")

# Print detailed results
for key, data in results.items():
    chain_name = key.replace('scc_', '').replace('_', ' ').title()
    print(f"\n{chain_name} Chain:")
    print(f"  Mean SCC: {np.mean(data['avg_scc']):.4f} ± {np.std(data['avg_scc']):.4f}")
    print(f"  Range: {data['min_scc']:.4f} - {data['max_scc']:.4f}")

print(f"\n🔍 FRAME ANALYSIS DIAGNOSIS:")
print(f"Expected: {len(universe.trajectory)} frames")
print(f"Got: {len(time_us)} frames") 
print(f"Sampling factor: {len(universe.trajectory)//len(time_us) if len(time_us) > 0 else 'N/A'}")

if len(time_us) < len(universe.trajectory):
    print(f"\n⚠️  WARNING: Only analyzing {len(time_us)} out of {len(universe.trajectory)} frames!")
    print(f"   This is likely due to internal sampling in lipyphilic SCC analysis.")
    print(f"   The analysis is still valid but covers {len(time_us)/len(universe.trajectory)*100:.1f}% of frames.")
