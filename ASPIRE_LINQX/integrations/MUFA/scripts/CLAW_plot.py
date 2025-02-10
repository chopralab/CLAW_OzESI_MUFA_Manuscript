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

class CLAWPlot:
    """
    A class to handle plotting and comparing lipid ratio data between CLAW sample data and manual data.
    """
    def __init__(self, manual_data: dict = None):
        """
        Initializes the CLAWPlot object with manual data.
        
        Parameters:
            manual_data (dict, optional): A dictionary containing manual canola data with keys such as
                                          'TG', 'Crude', 'Degummed', 'RBD'. If None, a default dataset is used.
        """
        if manual_data is None:
            manual_data = {
                'TG': ['TG(52:2)', 'TG(52:3)', 'TG(52:4)', 'TG(54:2)', 'TG(54:3)', 'TG(54:4)', 'TG(54:5)'],
                'Crude': [3.92, 2.2, 2.31, 4.27, 4.88, 3.78, 5.58],
                'Degummed': [3.76, 3.25, 2.25, 3.03, 4.94, 4.06, 5.65],
                'RBD': [4.13, 2.34, 2.18, 4.36, 4.57, 4.04, 4.45]
            }
        self.manual_df = pd.DataFrame(manual_data)

    def plot_ratio_comparison(self, df_sample_ratio: str, plot_title: str, caitlin_column: str):
        """
        Plots and compares the CLAW ratio data (from a CSV file) with the manual data.
        
        This method reads the sample ratio data from a CSV file whose path is provided as a string.
        It then plots:
            - The sample data (CLAW Ratio) using the 'Lipid' column (x-axis) and 'Ratio' column (y-axis).
            - The manual data (Caitlin data) using the 'TG' column (x-axis) and the specified 'caitlin_column' (y-axis)
              from the manual data provided at initialization.
        
        The plot is formatted with a title, axis labels, legend, and gridlines, and is saved as a PNG file
        (300 dpi) in the current directory with a filename based on the provided plot title.
        
        Parameters:
            df_sample_ratio (str): CSV file path for the sample ratio data.
            plot_title (str): Title for the plot (and used in the output filename).
            caitlin_column (str): Column name in the manual data to be plotted.
        """
        # Validate input types
        if not isinstance(df_sample_ratio, str):
            raise TypeError("df_sample_ratio must be a string representing the CSV file path")
        if not isinstance(plot_title, str):
            raise TypeError("plot_title must be a string")
        if not isinstance(caitlin_column, str):
            raise TypeError("caitlin_column must be a string")
        
        # Read the sample data from the CSV file
        df_sample = pd.read_csv(df_sample_ratio)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the sample data (CLAW Ratio) using a black line with circle markers
        plt.plot(df_sample['Lipid'], df_sample['Ratio'], label='CLAW Ratio',
                 color='black', marker='o', markersize=12, linewidth=2)
        
        # Plot the manual data (Caitlin data) using a blue line with circle markers
        plt.plot(self.manual_df['TG'], self.manual_df[caitlin_column],
                 label=f"Manual {plot_title}", color='blue', marker='o',
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
        output_filename = f'canola_paper_Crude.png'
        
        # Save the plot as a PNG file at 300 dpi resolution
        plt.savefig(output_filename, dpi=300)
        
        # Close the figure to free up resources
        plt.close()
        
        # Print a message confirming that the PNG has been saved
        print(f"PNG saved as '{output_filename}' in the directory.")

# ------------------------------------------------------------------------------
# Example usage:
# Uncomment and modify the paths as needed to use this class.
#
# if __name__ == "__main__":
#     # Instantiate the CLAWPlot class (optionally pass your own manual_data dict)
#     claw_plotter = CLAWPlot()
#     
#     # Generate the comparison plot by providing the sample CSV file path, plot title, and manual data column.
#     claw_plotter.plot_ratio_comparison('path_to_sample_ratio_data.csv', 'Sample Title', 'Crude')
# ------------------------------------------------------------------------------
