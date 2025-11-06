#!/usr/bin/env python3
"""
Script to plot S_v, S_a, S_cp, S_sp, S_temp variables for each section from a CSV file.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def plot_variables(csv_path, output_dir=None):
    """
    Plot S_v, S_a, S_cp, S_sp, S_temp for each section from the CSV file.
    Creates one plot per variable and saves them all in the output directory.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the data
    output_dir : str, optional
        Directory to save the output plots. If None, saves in the same directory as the CSV
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sort by Section name (natural sort to handle S1, S2, S10 etc.)
    df['Section_num'] = df['Section'].str.extract('(\d+)').astype(int)
    df = df.sort_values('Section_num')
    
    # Variables to plot
    variables = ['v_mm_per_yr','sigma_v','a_mm_per_yr2','sigma_a','t_v','t_a','R2_temp','cp_mm']
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of sections
    sections = df['Section'].values
    n_sections = len(sections)
    x = np.arange(n_sections)
    
    # Colors for each variable
    colors = {
        'v_mm_per_yr': '#1f77b4',
        'sigma_v': '#ff7f0e',
        'a_mm_per_yr2': '#2ca02c',
        'sigma_a': '#d62728',
        't_v': '#9467bd',
        't_a': '#9467bd',
        'R2_temp': '#9467bd',
        'cp_mm': '#9467bd'
    }
    
    # Variable titles
    titles = {
        'v_mm_per_yr': 'Long-term velocity (slope) of the detrended displacement time series (mm/yr)',
        'sigma_v': 'Uncertainty (standard error) of the velocity estimate (mm/yr)',
        'a_mm_per_yr2': 'Acceleration of the detrended series (mm/yr^2)',
        'sigma_a': 'Uncertainty (standard error) of the acceleration estimate (mm/yr^2)',
        't_v': 'Velocity t-statistic = absolute velocity divided by its standard error',
        't_a': 'Acceleration t-statistic = absolute acceleration divided by its standard error',
        'R2_temp': 'Fraction of the original displacement explained by the environmental model',
        'cp_mm': 'Estimated step magnitude in the recent window (last K days) (mm)'
    }
    

# S_v: How large and well-supported the long-term displacement trend is after detrending.
# S_a: How strongly the displacement is speeding up or slowing down after detrending.
# S_cp: How big and how recent the step-like change is in the latest analysis window.
# S_sp: How consistent the section’s recent motion is with its neighboring sections on the same span.
# S_temp: How much of the motion remains unexplained by temperature or seasonal effects.

    output_paths = []
    
    # Create a separate plot for each variable
    for var in variables:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get values for this variable
        values = df[var].values
        
        # Create bar plot
        bars = ax.bar(x, values, color=colors[var], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_xlabel('Section', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(titles[var], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sections, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        #ax.set_ylim([0, 1.05])  # Since all S_ variables appear to be normalized to [0, 1]
        
        # Add horizontal line at y=1 for reference
        #ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'{var}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        output_paths.append(output_path)
        
        # Close the figure to free memory
        plt.close()
    
    return output_paths


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Plot v_mm_per_yr, sigma_v, a_mm_per_yr2, sigma_a, t_v, t_a, R2_temp, cp_mm variables for each section from a CSV file. Creates one plot per variable.'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to the CSV file (e.g., /insar-data/Gerin/report/Gerin_bridge_risk.csv)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        dest='output_dir',
        help='Output directory for the plots (default: same directory as CSV)'
    )
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return 1
    
    # Generate the plots
    try:
        output_paths = plot_variables(args.csv_path, args.output_dir)
        print(f"\nSuccessfully created {len(output_paths)} plots")
        return 0
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

