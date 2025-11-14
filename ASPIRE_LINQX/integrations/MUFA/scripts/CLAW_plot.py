#!/usr/bin/env python3
"""
CLAWPlot.py

This module defines the CLAWPlot class which provides functionality to compare and
plot lipid ratio data from two data sources:
    - The CLAW (sample) data, which is read from a CSV file.
    - The manual (Caitlin) data, which can be provided at initialization.

The class method `plot_ratio_comparison` performs the following steps:
    1. Reads the sample data CSV file into a pandas DataFrame.
    2. Plots the sample data using the 'Lipid' column for the x-axis and the 'Ratio' column for the y-axis.
    3. Plots the manual data (Caitlin data) using the 'TG' column for the x-axis and a specified column for the y-axis.
    4. Formats the plot with a title, axis labels, a legend, and gridlines.
    5. Saves the plot as a PNG file in the current directory with a filename based on the provided plot title.
    6. Prints a confirmation message when the PNG file has been saved.

Example usage:
    # Instantiate the CLAWPlot class. Optionally, you can pass a custom manual_data dictionary.
    claw_plotter = CLAWPlot()
    
    # Create the comparison plot using a sample CSV file, plot title, and manual data column name.
    claw_plotter.plot_ratio_comparison('path_to_sample_ratio_data.csv', 'Sample Title', 'Crude')
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os

class CLAWPlot:
    """
    A class to handle plotting and comparing lipid ratio data between CLAW sample data and manual data.
    """
    def __init__(self, manual_data: dict = None, manual_csv: str = None):
        """
        Initializes the CLAWPlot object with manual data.
        
        Parameters:
            manual_data (dict, optional): A dictionary containing manual canola data with keys such as
                                          'TG', 'Crude', 'Degummed', 'RBD'. If None, a default dataset is used.
            manual_csv (str, optional): Path to CSV file containing manual data with 'Lipid' and 'n-9/n-7_Ratio' columns.
                                       If provided, this takes precedence over manual_data.
        """
        if manual_csv is not None:
            # Load manual data from CSV file
            self.manual_df = pd.read_csv(manual_csv)
        elif manual_data is None:
            manual_data = {
                'TG': ['TG(52:2)', 'TG(52:3)', 'TG(52:4)', 'TG(54:2)', 'TG(54:3)', 'TG(54:4)', 'TG(54:5)'],
                'Crude': [3.92, 2.2, 2.31, 4.27, 4.88, 3.78, 5.58],
                'Degummed': [3.76, 3.25, 2.25, 3.03, 4.94, 4.06, 5.65],
                'RBD': [4.13, 2.34, 2.18, 4.36, 4.57, 4.04, 4.45]
            }
            self.manual_df = pd.DataFrame(manual_data)
        else:
            self.manual_df = pd.DataFrame(manual_data)

    def plot_ratio_comparison(self, df_sample_ratio: str, plot_title: str, caitlin_column: str, ratio_type: str = 'intensity', output_dir: str = 'agent_plots'):
        """
        Plots and compares the CLAW ratio data (from a CSV file) with the manual data.
        
        This method reads the sample ratio data from a CSV file whose path is provided as a string.
        It then plots:
            - The sample data (CLAW Ratio) using the 'Lipid' column (x-axis) and the ratio column (y-axis).
            - The manual data (Caitlin data) using the 'TG' or 'Lipid' column (x-axis) and 'n-9/n-7_Ratio' (y-axis)
              from the manual data provided at initialization.
        
        The plot is formatted with a title, axis labels, legend, and gridlines, and is saved as a PNG file
        (300 dpi) in the specified directory with a filename based on the provided plot title.
        
        Parameters:
            df_sample_ratio (str): CSV file path for the sample ratio data.
            plot_title (str): Title for the plot (and used in the output filename).
            caitlin_column (str): Column name in the manual data to be plotted (e.g., 'Crude', 'Degummed', 'RBD').
            ratio_type (str): Type of ratio to use from CLAW data - 'area' for 'n-9/n-7_Ratio' or 'intensity' for 'n-9/n-7_Intensity_Ratio' (default: 'intensity').
            output_dir (str): Directory to save the plot (default: 'agent_plots').
        """
        # Validate input types
        if not isinstance(df_sample_ratio, str):
            raise TypeError("df_sample_ratio must be a string representing the CSV file path")
        if not isinstance(plot_title, str):
            raise TypeError("plot_title must be a string")
        if not isinstance(caitlin_column, str):
            raise TypeError("caitlin_column must be a string")
        
        # Validate ratio_type and determine the column name
        if ratio_type.lower() == 'area':
            ratio_column = 'n-9/n-7_Ratio'
        elif ratio_type.lower() == 'intensity':
            ratio_column = 'n-9/n-7_Intensity_Ratio'
        else:
            raise ValueError(f"ratio_type must be 'area' or 'intensity', got '{ratio_type}'")
        
        # Read the sample data from the CSV file
        df_sample = pd.read_csv(df_sample_ratio)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Determine label for CLAW data based on ratio type
        claw_label = f'CLAW {ratio_type.capitalize()}'
        
        # Plot the sample data (CLAW Ratio) using a black line with circle markers
        plt.plot(df_sample['Lipid'], df_sample[ratio_column], label=claw_label,
                 color='black', marker='o', markersize=12, linewidth=2)
        
        # Plot the manual data (always area-based) using a blue line with circle markers
        # Use 'TG' column if it exists, otherwise use 'Lipid' column
        x_col = 'TG' if 'TG' in self.manual_df.columns else 'Lipid'
        y_col = caitlin_column if caitlin_column in self.manual_df.columns else 'n-9/n-7_Ratio'
        plt.plot(self.manual_df[x_col], self.manual_df[y_col],
                 label=f"Manual Area", color='blue', marker='o',
                 markersize=12, linewidth=2)
        
        # Add title, labels, and legend to the plot
        plt.title(f'{plot_title} n-9 n-7 Ratios', fontsize=14)
        plt.xlabel('Lipid', fontsize=12)
        plt.ylabel('Ratio', fontsize=12)
        plt.legend()
        
        # Add grid, rotate x-axis labels for better readability, and adjust layout
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Define the output filename for the PNG file
        output_filename = f'canola_paper_{plot_title}.png'
        
        # If output directory is specified, create it and prepend to filename
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, output_filename)
        
        # Save the plot as a PNG file at 300 dpi resolution
        plt.savefig(output_filename, dpi=300)
        
        # Close the figure to free up resources
        plt.close()
        
        # Print a message confirming that the PNG has been saved
        print(f"PNG saved as '{output_filename}' in the directory.")

# ------------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate canola oil comparison plots between CLAW and manual data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CLAW_plot.py --claw results/df_CrudeCanola_O3on_150gN3_02082023_summary.csv \\
                      --manual results/df_manual_crude.csv \\
                      --title Crude
  
  python CLAW_plot.py --claw results/df_DegummedCanola_O3on_150gN3_02082023_summary.csv \\
                      --manual results/df_manual_degummed.csv \\
                      --title Degummed
        """
    )
    
    parser.add_argument(
        '--claw',
        required=True,
        help='Path to CLAW data CSV file (with Lipid and n-9/n-7_Intensity_Ratio columns)'
    )
    
    parser.add_argument(
        '--manual',
        required=True,
        help='Path to manual data CSV file (with Lipid and n-9/n-7_Ratio columns)'
    )
    
    parser.add_argument(
        '--title',
        required=True,
        help='Plot title (e.g., Crude, Degummed, RBD)'
    )
    
    parser.add_argument(
        '--ratio-type',
        choices=['area', 'intensity'],
        default='intensity',
        help="Type of ratio to use from CLAW data: 'area' for n-9/n-7_Ratio or 'intensity' for n-9/n-7_Intensity_Ratio (default: intensity)"
    )
    
    parser.add_argument(
        '--manual-ratio-col',
        default='n-9/n-7_Ratio',
        help='Column name for ratio in manual data (default: n-9/n-7_Ratio, always uses this)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='../agent_plots',
        help='Output directory for plots (default: ../agent_plots)'
    )
    
    args = parser.parse_args()
    
    # Create CLAWPlot instance with manual data
    print(f"Loading manual data from: {args.manual}")
    plotter = CLAWPlot(manual_csv=args.manual)
    
    # Generate the plot
    print(f"Generating plot for {args.title} using {args.ratio_type} ratios...")
    plotter.plot_ratio_comparison(
        df_sample_ratio=args.claw,
        plot_title=args.title,
        caitlin_column=args.manual_ratio_col,
        ratio_type=args.ratio_type,
        output_dir=args.output_dir
    )
    
    print(f"✓ Plot saved successfully!")
