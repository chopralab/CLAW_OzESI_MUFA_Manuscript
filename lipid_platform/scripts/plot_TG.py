import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
import re
from typing import Dict, Any, List, Tuple, Optional

# #Unique Sample IDs:
# CrudeCanola_O3on_150gN3_02082023
# DegummedCanola_O3on_150gN3_02082023
# RBDCanola_O3on_150gN3_02082023

# Define simplified default configuration dictionary
DEFAULT_CONFIG = {
    # Default colors for different db positions
    'colors': {
        'n-7': 'blue',
        'n-9': 'red',
        'default': 'green'
    },
    
    # Default relative heights for peak width calculation (lower = wider peaks)
    'rel_heights': {
        'n-7': 0.5,
        'n-9': 0.75,
        'default': 0.6
    },
    
    # Default width adjustment factors for visualization
    'width_factors': {
        'n-7': 0.75,  # Shrink n-7 by 25%
        'n-9': 1.5,   # Expand n-9 by 50%
        'default': 1.0
    }
}

def analyze_lipid_peaks_with_peakfinder(df, lipid_pattern, sample_id, config=None, show_plots=False):
    """
    Analyze chromatogram peaks for a specific lipid pattern using SciPy's peak finder.
    Only uses the peak with the largest OzESI_Intensity value for each position (n-9 and n-7)
    for the ratio calculation.
    Customizable width parameters based on the provided configuration.
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe containing lipid data
    lipid_pattern : str
        The lipid pattern to analyze (regex)
    sample_id : str
        The sample ID to analyze
    config : dict, optional
        Configuration dictionary for customizing analysis
    show_plots : bool, optional
        Whether to display the plots (default: False)
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Filter the dataframe to match the criteria
    filtered_df = df[
        (df['Lipid'].str.contains(lipid_pattern, regex=True)) & 
        (df['Sample_ID'] == sample_id)
    ]
    
    # Check if we have data
    if filtered_df.empty:
        print(f"No data found for lipid pattern '{lipid_pattern}' in sample '{sample_id}'")
        return {}
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    peak_data = {}
    
    # Get unique positions from the data
    positions = list(filtered_df['db_pos'].unique())
    
    if not positions:
        print(f"No db_pos values found for lipid pattern '{lipid_pattern}' in sample '{sample_id}'")
        return {}
    
    # Process each position (n-7 and n-9)
    for pos in positions:
        pos_data = filtered_df[filtered_df['db_pos'] == pos]
        
        if not pos_data.empty:
            # Sort by retention time to ensure correct peak finding
            pos_data = pos_data.sort_values('Retention_Time')
            
            # Extract arrays for processing
            retention_times = pos_data['Retention_Time'].values
            intensities = pos_data['OzESI_Intensity'].values
            
            # Get color for this position
            if pos in config['colors']:
                color = config['colors'][pos]
            else:
                color = config['colors']['default']
            
            # Plot the raw data
            ax.plot(
                retention_times, 
                intensities, 
                label=f"{lipid_pattern} {pos}",
                color=color,
                marker='o',
                linestyle='-',
                markersize=4
            )
            
            # Find peaks in the chromatogram
            peaks, properties = find_peaks(
                intensities, 
                height=0.5 * np.max(intensities), 
                distance=3, 
                prominence=0.3 * np.max(intensities)
            )
            
            if len(peaks) > 0:
                # Initialize peak data structure
                peak_data[pos] = {
                    'all_peak_indices': peaks,
                    'all_peak_heights': properties['peak_heights'],
                    'all_peak_areas': [],
                    'all_retention_times': retention_times[peaks],
                    'largest_peak_index': None,
                    'largest_peak_height': 0,
                    'largest_peak_area': 0,
                    'largest_peak_rt': 0
                }
                
                # Get the relative height for this position
                if pos in config['rel_heights']:
                    rel_height = config['rel_heights'][pos]
                else:
                    rel_height = config['rel_heights']['default']
                
                # Find peak widths using the configured relative height
                widths, width_heights, left_ips, right_ips = peak_widths(
                    intensities, peaks, rel_height=rel_height
                )
                
                # Get the width adjustment factor for this position
                if pos in config['width_factors']:
                    width_factor = config['width_factors'][pos]
                else:
                    width_factor = config['width_factors']['default']
                
                # Find the largest peak
                largest_peak_idx = None
                largest_intensity = 0
                
                # For each detected peak, calculate and visualize its area
                for i, peak_idx in enumerate(peaks):
                    # Calculate bounds for integration
                    left_idx = int(left_ips[i])
                    right_idx = int(right_ips[i])
                    
                    # Ensure indices are within bounds
                    left_idx = max(0, left_idx)
                    right_idx = min(len(intensities) - 1, right_idx)
                    
                    # Get the x and y values for this peak (for area calculation)
                    peak_x = retention_times[left_idx:right_idx+1]
                    peak_y = intensities[left_idx:right_idx+1]
                    
                    # Calculate area 
                    peak_area = np.trapz(peak_y, peak_x)
                    peak_data[pos]['all_peak_areas'].append(peak_area)
                    
                    # Check if this is the largest peak by intensity
                    if intensities[peak_idx] > largest_intensity:
                        largest_intensity = intensities[peak_idx]
                        largest_peak_idx = i
                        peak_data[pos]['largest_peak_index'] = peak_idx
                        peak_data[pos]['largest_peak_height'] = intensities[peak_idx]
                        peak_data[pos]['largest_peak_area'] = peak_area
                        peak_data[pos]['largest_peak_rt'] = retention_times[peak_idx]
                    
                    # Determine color and alpha based on whether this is the largest peak
                    fill_color = color
                    fill_alpha = 0.1  # Less visible for non-largest peaks
                    line_style = ':'  # Dotted line for non-largest peaks
                    
                    # Plot all peaks with less emphasis
                    # Adjust the width for visualization
                    peak_center_idx = peak_idx
                    peak_width = right_idx - left_idx
                    adjusted_width = int(peak_width * width_factor / 2)
                    vis_left_idx = max(0, peak_center_idx - adjusted_width)
                    vis_right_idx = min(len(intensities) - 1, peak_center_idx + adjusted_width)
                    vis_x = retention_times[vis_left_idx:vis_right_idx+1]
                    vis_y = intensities[vis_left_idx:vis_right_idx+1]
                    
                    # Fill the area for visualization with low alpha
                    ax.fill_between(vis_x, vis_y, alpha=fill_alpha, color=fill_color)
                    ax.axvline(x=retention_times[peak_idx], color=color, linestyle=line_style, alpha=0.3)
                    
                    # Annotate all peaks with small, light text
                    ax.annotate(
                        f"Area: {peak_area:.2f}",
                        xy=(retention_times[peak_idx], intensities[peak_idx]),
                        xytext=(retention_times[peak_idx], intensities[peak_idx] * 1.1),
                        fontsize=7,
                        ha='center',
                        alpha=0.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.1)
                    )
                
                # Now highlight the largest peak more prominently
                if largest_peak_idx is not None:
                    peak_idx = peaks[largest_peak_idx]
                    left_idx = int(left_ips[largest_peak_idx])
                    right_idx = int(right_ips[largest_peak_idx])
                    
                    # Ensure indices are within bounds
                    left_idx = max(0, left_idx)
                    right_idx = min(len(intensities) - 1, right_idx)
                    
                    # Adjust the width for visualization
                    peak_center_idx = peak_idx
                    peak_width = right_idx - left_idx
                    adjusted_width = int(peak_width * width_factor / 2)
                    vis_left_idx = max(0, peak_center_idx - adjusted_width)
                    vis_right_idx = min(len(intensities) - 1, peak_center_idx + adjusted_width)
                    vis_x = retention_times[vis_left_idx:vis_right_idx+1]
                    vis_y = intensities[vis_left_idx:vis_right_idx+1]
                    
                    # Highlight the largest peak
                    ax.fill_between(vis_x, vis_y, alpha=0.5, color=color)
                    ax.axvline(x=retention_times[peak_idx], color=color, linestyle='-', alpha=0.8)
                    
                    # Annotate the largest peak with bold text
                    ax.annotate(
                        f"LARGEST PEAK\nArea: {peak_data[pos]['largest_peak_area']:.2f}",
                        xy=(retention_times[peak_idx], intensities[peak_idx]),
                        xytext=(retention_times[peak_idx], intensities[peak_idx] * 1.2),
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.5)
                    )
            else:
                print(f"No peaks detected for {pos}. Try adjusting peak detection parameters.")
                peak_data[pos] = {'largest_peak_area': 0}
    
    # Calculate the ratio based only on the largest peaks
    if 'n-9' in peak_data and 'n-7' in peak_data and peak_data['n-7']['largest_peak_area'] > 0:
        ratio = peak_data['n-9']['largest_peak_area'] / peak_data['n-7']['largest_peak_area']
        ratio_text = f"n-9/n-7 Ratio (largest peaks only): {ratio:.4f}"
        ax.annotate(
            ratio_text,
            xy=(0.5, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3)
        )
        peak_data['ratio'] = ratio
    else:
        ratio_text = "Cannot calculate ratio (missing peaks or zero area)"
        print(ratio_text)
        peak_data['ratio'] = None
    
    # Set plot title and labels
    ax.set_title(f"Chromatogram Peak Analysis for {lipid_pattern} in {sample_id}\n(Using largest peak per position for ratio)")
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("OzESI Intensity")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot only if requested
    if show_plots:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    # Print the results
    print(f"\nPeak Analysis for {lipid_pattern} in {sample_id}:")
    print("-" * 50)
    for pos in peak_data:
        if pos not in ['ratio', 'ratio_description']:
            if 'largest_peak_area' in peak_data[pos]:
                print(f"{pos} Largest Peak Area: {peak_data[pos]['largest_peak_area']:.2f}")
                if 'largest_peak_rt' in peak_data[pos]:
                    print(f"  Retention Time: {peak_data[pos]['largest_peak_rt']:.2f}")
                print(f"  Peak Height: {peak_data[pos].get('largest_peak_height', 0):.2f}")
            
            if 'all_peak_areas' in peak_data[pos]:
                print(f"  All {len(peak_data[pos]['all_peak_areas'])} peaks:")
                for i, area in enumerate(peak_data[pos]['all_peak_areas']):
                    rt = peak_data[pos]['all_retention_times'][i]
                    height = peak_data[pos]['all_peak_heights'][i]
                    print(f"    Peak {i+1} - RT: {rt:.2f}, Height: {height:.2f}, Area: {area:.2f}")
    
    if peak_data.get('ratio') is not None:
        print(f"\n{ratio_text}")
    else:
        print(f"\nUnable to calculate ratio (missing peaks or zero area)")
    
    return peak_data

def analyze_all_lipids_for_sample(df, sample_id, config=None, lipid_patterns=None, show_plots=False):
    """
    Analyze all unique lipid patterns for a specific sample ID.
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe containing lipid data
    sample_id : str
        The sample ID to analyze
    config : dict, optional
        Configuration dictionary for customizing analysis
    lipid_patterns : list, optional
        List of lipid patterns to analyze. If None, extracts from data
    show_plots : bool, optional
        Whether to display the plots (default: False)
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Filter dataframe by sample ID
    sample_df = df[df['Sample_ID'] == sample_id]
    
    if sample_df.empty:
        print(f"No data found for sample '{sample_id}'")
        return {}
    
    # If lipid_patterns is not provided, extract them from the data
    if lipid_patterns is None:
        # Extract TG patterns like TG(52:3) from the Lipid column
        lipid_patterns = []
        for lipid in sample_df['Lipid'].unique():
            # Extract the TG pattern using regex
            match = re.search(r'\[(TG\(\d+:\d+\))\]', lipid)
            if match:
                lipid_patterns.append(match.group(1))
        
        # Remove duplicates
        lipid_patterns = sorted(list(set(lipid_patterns)))
    
    print(f"Found {len(lipid_patterns)} unique lipid patterns in sample {sample_id}:")
    for pattern in lipid_patterns:
        print(f"  - {pattern}")
    
    # Store results for each lipid pattern
    all_results = {}
    
    # Analyze each lipid pattern
    for pattern in lipid_patterns:
        print(f"\n{'-' * 60}")
        print(f"Analyzing {pattern}...")
        pattern_regex = pattern.replace("(", "\\(").replace(")", "\\)")
        results = analyze_lipid_peaks_with_peakfinder(df, pattern_regex, sample_id, config, show_plots=show_plots)
        all_results[pattern] = results
    
    # Summarize all results
    print("\n" + "=" * 80)
    print(f"SUMMARY OF ALL LIPID PATTERNS FOR SAMPLE {sample_id}")
    print("=" * 80)
    
    # Create a data frame to store the summary
    summary_data = []
    
    for lipid, data in all_results.items():
        n7_area = data.get('n-7', {}).get('total_area', 0)
        n9_area = data.get('n-9', {}).get('total_area', 0)
        ratio = data.get('ratio', None)
        
        summary_data.append({
            'Lipid': lipid,
            'n-7 Area': n7_area,
            'n-9 Area': n9_area,
            'n-9/n-7 Ratio': ratio
        })
    
    # Create and display the summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, na_rep='N/A'))
    
    return all_results

def visualize_lipid_ratios(all_results, sample_id, show_plots=False):
    """
    Create a bar chart visualization of n-9/n-7 ratios across all lipids.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of analysis results for each lipid
    sample_id : str
        The sample ID being analyzed
    show_plots : bool, optional
        Whether to display the plots (default: False)
    """
    lipids = []
    ratios = []
    
    for lipid, data in all_results.items():
        if data.get('ratio') is not None:
            lipids.append(lipid)
            ratios.append(data['ratio'])
    
    if lipids:
        plt.figure(figsize=(14, 8))
        bars = plt.bar(lipids, ratios, color='skyblue')
        
        # Add ratio values on top of each bar
        for bar, ratio in zip(bars, ratios):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f'{ratio:.4f}',
                ha='center', 
                rotation=0,
                fontsize=9
            )
        
        plt.title(f'n-9/n-7 Ratios Across All Lipids in {sample_id}')
        plt.xlabel('Lipid Pattern')
        plt.ylabel('n-9/n-7 Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("No valid ratios found to visualize.")

def create_custom_config():
    """
    Create a custom configuration with specific settings for different lipids.
    """
    # Start with the default config
    custom_config = DEFAULT_CONFIG.copy()
    
    # We need to make a deep copy of the nested dictionaries
    custom_config['colors'] = custom_config['colors'].copy()
    custom_config['rel_heights'] = custom_config['rel_heights'].copy()
    custom_config['width_factors'] = custom_config['width_factors'].copy()
    

    # #Crude Canola
    # custom_config['lipid_specific'] = {
    #     'TG(52:2)': {
    #         'n-7': {'rel_height': 0.5},
    #         'n-9': {'rel_height': 0.7}
    #     },
    #     'TG(52:3)': {
    #         'n-7': {'rel_height': 0.5},
    #         'n-9': {'rel_height': 0.75}
    #     },
    #     'TG(52:4)': {
    #         'n-7': {'rel_height': 0.4},
    #         'n-9': {'rel_height': 0.7}
    #     },
    #     'TG(54:2)': {
    #         'n-7': {'rel_height': 0.55},
    #         'n-9': {'rel_height': 0.8}
    #     },
    #     'TG(54:3)': {
    #         'n-7': {'rel_height': 0.8},
    #         'n-9': {'rel_height': 0.8}
    #     },
    #     'TG(54:4)': {
    #         'n-7': {'rel_height': 0.7},
    #         'n-9': {'rel_height': 0.8}
    #     },
    #     'TG(54:5)': {
    #         'n-7': {'rel_height': 0.4},
    #         'n-9': {'rel_height': 0.5}
    #     }
    # }

    # #Degummed Canola
    # custom_config['lipid_specific'] = {
    #     'TG(52:2)': {
    #         'n-7': {'rel_height': 0.5},
    #         'n-9': {'rel_height': 0.7}
    #     },
    #     'TG(52:3)': {
    #         'n-7': {'rel_height': 0.5},
    #         'n-9': {'rel_height': 0.7}
    #     },
    #     'TG(52:4)': {
    #         'n-7': {'rel_height': 0.4},
    #         'n-9': {'rel_height': 0.8}
    #     },
    #     'TG(54:2)': {
    #         'n-7': {'rel_height': 0.4},
    #         'n-9': {'rel_height': 0.75}
    #     },
    #     'TG(54:3)': {
    #         'n-7': {'rel_height': 0.8},
    #         'n-9': {'rel_height': 0.8}
    #     },
    #     'TG(54:4)': {
    #         'n-7': {'rel_height': 0.7},
    #         'n-9': {'rel_height': 0.8}
    #     },
    #     'TG(54:5)': {
    #         'n-7': {'rel_height': 0.5},
    #         'n-9': {'rel_height': 0.65}
    #     }
    # }

    #RBDCanola
    custom_config['lipid_specific'] = {
        'TG(52:2)': {
            'n-7': {'rel_height': 0.55},
            'n-9': {'rel_height': 0.6}
        },
        'TG(52:3)': {
            'n-7': {'rel_height': 0.5},
            'n-9': {'rel_height': 0.7}
        },
        'TG(52:4)': {
            'n-7': {'rel_height': 0.45},
            'n-9': {'rel_height': 0.55}
        },
        'TG(54:2)': {
            'n-7': {'rel_height': 0.4},
            'n-9': {'rel_height': 0.8}
        },
        'TG(54:3)': {
            'n-7': {'rel_height': 0.9},
            'n-9': {'rel_height': 0.95}
        },
        'TG(54:4)': {
            'n-7': {'rel_height': 0.85},
            'n-9': {'rel_height': 0.95}
        },
        'TG(54:5)': {
            'n-7': {'rel_height': 0.4},
            'n-9': {'rel_height': 0.7}
        }
    }

    return custom_config


import copy
import re

# 1) Optional: detect sample type from the sample_id string
def infer_sample_type(sample_id: str) -> str:
    sid = sample_id.lower()
    if "crude" in sid:
        return "Crude"
    if "degummed" in sid:
        return "Degummed"
    if "rbd" in sid:
        return "RBD"
    return "Default"  # fallback

# 2) Define lipid-specific presets per sample type (put your tuned values here)
SAMPLE_PRESETS = {
    "Crude": {
        "lipid_specific": {
            "TG(52:2)": {"n-7": {"rel_height": 0.5},  "n-9": {"rel_height": 0.7}},
            "TG(52:3)": {"n-7": {"rel_height": 0.5},  "n-9": {"rel_height": 0.75}},
            "TG(52:4)": {"n-7": {"rel_height": 0.4},  "n-9": {"rel_height": 0.7}},
            "TG(54:2)": {"n-7": {"rel_height": 0.55}, "n-9": {"rel_height": 0.8}},
            "TG(54:3)": {"n-7": {"rel_height": 0.8},  "n-9": {"rel_height": 0.8}},
            "TG(54:4)": {"n-7": {"rel_height": 0.7},  "n-9": {"rel_height": 0.8}},
            "TG(54:5)": {"n-7": {"rel_height": 0.4},  "n-9": {"rel_height": 0.5}},
        }
    },
    "Degummed": {
        "lipid_specific": {
            "TG(52:2)": {"n-7": {"rel_height": 0.5},  "n-9": {"rel_height": 0.7}},
            "TG(52:3)": {"n-7": {"rel_height": 0.5},  "n-9": {"rel_height": 0.7}},
            "TG(52:4)": {"n-7": {"rel_height": 0.4},  "n-9": {"rel_height": 0.8}},
            "TG(54:2)": {"n-7": {"rel_height": 0.4},  "n-9": {"rel_height": 0.75}},
            "TG(54:3)": {"n-7": {"rel_height": 0.8},  "n-9": {"rel_height": 0.8}},
            "TG(54:4)": {"n-7": {"rel_height": 0.7},  "n-9": {"rel_height": 0.8}},
            "TG(54:5)": {"n-7": {"rel_height": 0.5},  "n-9": {"rel_height": 0.65}},
        }
    },
    "RBD": {
        "lipid_specific": {
            "TG(52:2)": {"n-7": {"rel_height": 0.55}, "n-9": {"rel_height": 0.6}},
            "TG(52:3)": {"n-7": {"rel_height": 0.5},  "n-9": {"rel_height": 0.7}},
            "TG(52:4)": {"n-7": {"rel_height": 0.45}, "n-9": {"rel_height": 0.55}},
            "TG(54:2)": {"n-7": {"rel_height": 0.4},  "n-9": {"rel_height": 0.8}},
            "TG(54:3)": {"n-7": {"rel_height": 0.9},  "n-9": {"rel_height": 0.95}},
            "TG(54:4)": {"n-7": {"rel_height": 0.85}, "n-9": {"rel_height": 0.95}},
            "TG(54:5)": {"n-7": {"rel_height": 0.4},  "n-9": {"rel_height": 0.7}},
        }
    },
    "Default": {
        "lipid_specific": {}  # no overrides
    }
}

# 3) Safe deep-merge (preserves nested dicts)
def deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

# 4) Create a config for a given sample_id (parameterized factory)
def create_custom_config_for_sample(sample_id: str) -> dict:
    sample_type = infer_sample_type(sample_id)   # "Crude" | "Degummed" | "RBD" | "Default"

    # Start from a deep copy of DEFAULT_CONFIG
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # (Optional) ensure nested dicts exist
    cfg.setdefault("colors", {})
    cfg.setdefault("rel_heights", {})
    cfg.setdefault("width_factors", {})
    cfg.setdefault("lipid_specific", {})

    # Merge in the sample-specific overrides
    preset = SAMPLE_PRESETS.get(sample_type, SAMPLE_PRESETS["Default"])
    cfg = deep_update(cfg, preset)

    return cfg


def apply_lipid_specific_config(config, lipid_pattern):
    """
    Apply lipid-specific configuration for a given lipid pattern.
    
    This creates a temporary config with overridden values for the specific lipid.
    """
    # Create a new config that's a copy of the input config
    temp_config = {
        'colors': config['colors'].copy(),
        'rel_heights': config['rel_heights'].copy(),
        'width_factors': config['width_factors'].copy()
    }
    
    # Check if we have specific settings for this lipid
    if 'lipid_specific' in config and lipid_pattern in config['lipid_specific']:
        lipid_config = config['lipid_specific'][lipid_pattern]
        
        # Apply settings for each position
        for pos, pos_config in lipid_config.items():
            if 'color' in pos_config:
                temp_config['colors'][pos] = pos_config['color']
            if 'width_factor' in pos_config:
                temp_config['width_factors'][pos] = pos_config['width_factor']
            if 'rel_height' in pos_config:
                temp_config['rel_heights'][pos] = pos_config['rel_height']
    
    return temp_config

def main(show_plots=False):
    """
    Main function to run the analysis with custom config.
    
    Parameters:
    -----------
    show_plots : bool, optional
        Whether to display the plots (default: False)
    """
    # Set the sample ID
    sample_id = "RBDCanola_O3on_150gN3_02082023"
    
    # Create a custom configuration
    custom_config = create_custom_config()
    
    # File name to save final summary DataFrame
    save_df = 'df_RBD.csv'
    
    # Option 1: Analyze all lipid patterns in the sample
    all_results = {}
    
    # Filter dataframe by sample ID
    sample_df = df_canola_lipids[df_canola_lipids['Sample_ID'] == sample_id]
    
    # Extract lipid patterns from the data
    lipid_patterns = []
    for lipid in sample_df['Lipid'].unique():
        match = re.search(r'\[(TG\(\d+:\d+\))\]', lipid)
        if match:
            lipid_patterns.append(match.group(1))
    
    # Remove duplicates
    lipid_patterns = sorted(list(set(lipid_patterns)))
    
    print(f"Found {len(lipid_patterns)} unique lipid patterns in sample {sample_id}:")
    for pattern in lipid_patterns:
        print(f"  - {pattern}")
    
    # Analyze each lipid pattern with its specific configuration
    for pattern in lipid_patterns:
        print(f"\n{'-' * 60}")
        print(f"Analyzing {pattern}...")
        pattern_regex = pattern.replace("(", "\\(").replace(")", "\\)")
        
        # Apply lipid-specific configuration
        lipid_config = apply_lipid_specific_config(custom_config, pattern)
        
        # Analyze this specific lipid
        results = analyze_lipid_peaks_with_peakfinder(
            df_canola_lipids, pattern_regex, sample_id, lipid_config, show_plots=show_plots
        )
        all_results[pattern] = results
    
    # Summarize all results
    print("\n" + "=" * 80)
    print(f"SUMMARY OF ALL LIPID PATTERNS FOR SAMPLE {sample_id}")
    print("=" * 80)
    
    # Create a data frame to store the summary
    summary_data = []
    
    for lipid, data in all_results.items():
        n7_area = data.get('n-7', {}).get('largest_peak_area', 0)
        n9_area = data.get('n-9', {}).get('largest_peak_area', 0)
        ratio = data.get('ratio', None)
        
        summary_data.append({
            'Lipid': lipid,
            'n-7 Area': n7_area,
            'n-9 Area': n9_area,
            'n-9/n-7 Ratio': ratio
        })
    
    # Create and display the summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, na_rep='N/A'))

    # Save to CSV
    summary_df.to_csv(save_df, index=False)
    print(f"\nSaved summary results to: {save_df}")
    
    # Visualize the ratios
    visualize_lipid_ratios(all_results, sample_id, show_plots=show_plots)
    
    return all_results

# If this script is run directly (not imported), run the main function
if __name__ == "__main__":
    results = main()