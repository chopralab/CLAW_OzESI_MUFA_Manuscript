import pandas as pd
import plotly.express as px

def plot_ratio(df, color_mapping, output_directory, ratio_threshold=None):
    """
    Plots the ratio of lipids for each unique Sample_ID in the given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns 'Sample_ID', 'Lipid', and 'ratio'.
        color_mapping (dict): Mapping of patterns to colors for the Lipid values.
        output_directory (str): Directory where to save the plot images.
        ratio_threshold (float, optional): Minimum ratio value for plotting. If provided, rows with a ratio 
                                           value below this threshold will be excluded from plotting.

    Returns:
        None
    """
    import plotly.express as px
    import os

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Apply the ratio threshold filter if provided
    if ratio_threshold is not None:
        df = df[df['Ratio'] >= ratio_threshold]

    # Get the unique Sample_IDs
    sample_ids = df['Sample_ID'].unique()

    # Loop over the unique Sample_IDs
    for sample_id in sample_ids:

        # Filter the dataframe for the current Sample_ID
        df_sample = df[df['Sample_ID'] == sample_id]

        # Assign colors to Lipids based on patterns
        lipid_colors = []
        for lipid in df_sample['Lipid']:
            color = 'gray'  # Default color
            for pattern, pattern_color in color_mapping.items():
                if pattern in lipid:
                    color = pattern_color
                    break
            lipid_colors.append(color)

        # Create the bar plot
        fig = px.bar(df_sample, x='Lipid', y='Ratio', text='Ratio', title=f'Bar Plot for Sample_ID: {sample_id}')

        # Apply colors to the bars
        fig.update_traces(
            marker_color=lipid_colors,
            texttemplate='%{text:.2f}',
            textposition='auto',
            marker_line_width=0
        )

        # Customize the layout
        fig.update_layout(
            uniformtext_minsize=18,
            uniformtext_mode='hide',
            xaxis=dict(
                title='Lipid',
                titlefont=dict(size=16)
            ),
            yaxis=dict(
                title='Ratio',
                titlefont=dict(size=16),
                tickfont=dict(size=16)  # Set the font size of y-axis labels
            ),
            title=dict(
                text=f'Sample_ID: {sample_id}',
                font=dict(size=20)  # Set the title font size
            )
        )
        
        # Save the plot as an image
        file_name = os.path.join(output_directory, f"plot_{sample_id}.png")

        # Check if the file already exists
        index = 1
        while os.path.exists(file_name):
            file_name = os.path.join(output_directory, f"plot_{sample_id}_{index}.png")
            index += 1

        fig.write_image(file_name)




def printed_ratio(df_OzESI_ratio_sort):
    """
    Prints the Lipid, Sample_ID, db_pos, and ratio for each row in the given DataFrame.

    Parameters:
        df_OzESI_ratio_sort (pd.DataFrame): DataFrame with columns 'Lipid', 'Sample_ID', 'db_pos', and 'ratio'.

    Returns:
        None
    """
    # Iterate through each row in the DataFrame
    for index, row in df_OzESI_ratio_sort.iterrows():
        # Extract Lipid, Sample_ID, Labels and ratio from the row
        lipid = row['Lipid']
        sample_id = row['Sample_ID']
        db_pos = row['db_pos']
        ratio = row['Ratio']

        # Check if ratio is not NaN
        if not pd.isna(ratio):
            # Print out the values
            print(f'Lipid: {lipid}, Sample_ID: {sample_id}, db_pos: {db_pos}, Ratio: {ratio}')
            



import os
import pandas as pd
import plotly.graph_objects as go

def plot_chromatogram(file_path, plot_path, plot_name, x_range=None):
    # Read the CSV file
    data = pd.read_csv(file_path, skiprows=1) # Skipping the first row with metadata

    # If an x_range is provided, filter the data accordingly
    if x_range:
        start_time, end_time = x_range
        data = data[(data['X(Minutes)'] >= start_time) & (data['X(Minutes)'] <= end_time)]

    # Create the plot
    fig = go.Figure()

    # Add the line trace
    fig.add_trace(go.Scatter(x=data['X(Minutes)'], y=data['Y(Counts)'], mode='lines', name='Intensity'))

    # Set the title and axis labels
    fig.update_layout(
        title="Chromatogram of Canola Oil Crude Sample",
        xaxis_title="Time (minutes)",
        yaxis_title="Intensity (Counts)",
        font=dict(
            family="Arial, sans-serif",
            size=18
        ),
        showlegend=False
    )

    # Customize the plot appearance
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', linecolor='Black', linewidth=2, mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', linecolor='Black', linewidth=2, mirror=True,
                    exponentformat='e', showexponent='all')


    # Check if the plot path exists, if not then create it
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Check if the file exists, if it does add 1 and try again
    count = 1
    filename = plot_name
    while os.path.exists(f"{plot_path}{filename}"):
        filename = f"{plot_name[:-4]}_{count}{plot_name[-4:]}"
        count += 1

    # Save the image
    fig.write_image(f"{plot_path}{filename}", scale=2)

    # # Show the plot
    # fig.show()


#### Comparison Plot CLAW vs Manual

import pandas as pd
import matplotlib.pyplot as plt

# Manual Canola data for comparison
manual_canola = {
    'TG': ['TG(52:2)_FA18:1', 'TG(52:3)_FA18:1', 'TG(52:4)_FA18:1', 'TG(54:2)_FA18:1', 'TG(54:3)_FA18:1', 'TG(54:4)_FA18:1', 'TG(54:5)_FA18:1'],
    'Crude': [3.92, 2.2, 2.31, 4.27, 4.88, 3.78, 5.58],
    'Degummed': [3.76, 3.25, 2.25, 3.03, 4.94, 4.06, 5.65],
    'RBD': [4.13, 2.34, 2.18, 4.36, 4.57, 4.04, 4.45]
}

# Creating DataFrame for Manual Canola data
manual_df = pd.DataFrame(manual_canola)

def plot_ratio_comparison(df_sample_ratio, plot_title, caitlin_column):
    plt.figure(figsize=(10, 6))

    # Append _FA18:1 to each lipid name
    df_sample_ratio = df_sample_ratio.copy()
    df_sample_ratio['Lipid'] = df_sample_ratio['Lipid'].astype(str) + '_FA18:1'

    # Plotting sample data (Black line, circle markers)
    plt.plot(
        df_sample_ratio['Lipid'], 
        df_sample_ratio['Ratio'], 
        label='CLAW Max Intensity', 
        color='black', 
        marker='o', 
        markersize=14, 
        linewidth=2
    )
    
    # Plotting Manual data (Blue line, circle markers)
    plt.plot(
        manual_df['TG'], 
        manual_df[caitlin_column], 
        label="Manual Area", 
        color='blue', 
        marker='o', 
        markersize=14, 
        linewidth=2
    )
    
    # Labels and title
    plt.xlabel('Lipid', fontsize=20)
    plt.ylabel('n-9 / n-7 Ratio', fontsize=20)
    plt.legend(fontsize=16, loc='lower right')  # ← bottom right
    
    # Grid, ticks, and layout
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    
    # Save both PNG and PDF at 600 dpi
    save_base = f'Projects/canola/plots/{plot_title} n-9 n-7 Ratios'
    plt.savefig(f'{save_base}.png', dpi=600)
    plt.savefig(f'{save_base}.pdf', dpi=600)
    plt.show()


##### barplot paper
import pandas as pd
import plotly.graph_objects as go

def plot_canola_comparison(df_sample1_ratio, df_sample2_ratio, df_sample3_ratio, output_file=None):
    # First dataframe with renamed columns (CLAW for Crude, Degummed, and RBD)
    CLAW_canola = {
        'TG': ['TG(52:2)', 'TG(52:3)', 'TG(52:4)', 'TG(54:2)', 'TG(54:3)', 'TG(54:4)', 'TG(54:5)'],
        'Crude': df_sample1_ratio['Ratio'],  # Using the Crude data (df_sample1_ratio)
        'Degummed': df_sample2_ratio['Ratio'],  # Using the Degummed data (df_sample2_ratio)
        'RBD': df_sample3_ratio['Ratio']  # Using the RBD data (df_sample3_ratio)
    }

    # Create DataFrame for CLAW data
    CLAW_df = pd.DataFrame(CLAW_canola)

    # Create a grouped horizontal bar chart with gradient colors using Plotly
    fig = go.Figure(data=[
           go.Bar(name='Crude', y=CLAW_df['TG'], x=CLAW_df['Crude'], orientation='h',
                  marker=dict(color='blue', line=dict(color='darkblue', width=1), colorscale='Blues')),
           go.Bar(name='Degummed', y=CLAW_df['TG'], x=CLAW_df['Degummed'], orientation='h',
                  marker=dict(color='red', line=dict(color='darkred', width=1), colorscale='Reds')),
           go.Bar(name='RBD', y=CLAW_df['TG'], x=CLAW_df['RBD'], orientation='h',
                  marker=dict(color='LimeGreen', line=dict(color='darkgreen', width=1), colorscale='Greens'))
    ])

    # Change the bar mode to 'group'
    fig.update_layout(barmode='group',
                      title='<b>' + "Canola Oil TG Comparison Across Processing Steps" + '<b>',
                      title_x=0.5,
                      yaxis_title='<i>' + "Lipid Structure" + '<i>',
                      xaxis_title='<i>' + "n-9/n-7 FA 18:1 Isomer Ratio" + '<i>',
                      title_font_size=32,
                      yaxis_title_font=dict(size=24),
                      xaxis_title_font=dict(size=24),
                      width=1000,
                      height=600,
                      legend_title_font=dict(size=24),
                      legend_font=dict(size=20),
                      legend_traceorder='reversed')

    # Update y-axis to reverse the order of the categories (from 52:2 to 54:5)
    fig.update_yaxes(categoryorder='category ascending', tickfont=dict(family="Arial Black", size=20))
    fig.update_xaxes(tickfont=dict(family="Arial Black", size=20))

    # Show the figure or save to a file
    #save as pdf
    fig.write_image("Projects/canola/plots/bar/CLAW_MUFA_Barplot.pdf", scale=2)
 
        # Show the figure or save to a file
    if output_file is not None:
        fig.write_image(output_file)
    else:
        fig.show()






#### bar plot to see all lipids in canola per sample
import os
import re

# Function to create sorting keys for lipids
def lipid_sort_key(lipid):
    matches = re.findall(r'(\d+)', lipid)
    if len(matches) >= 2:
        return (int(matches[0]), int(matches[1]))
    elif len(matches) == 1:
        return (int(matches[0]), 0)
    else:
        return (0, 0)

# Define color mappings for Lipid patterns
color_mapping = {
    '50': 'red',
    '51': 'brown',
    '52': 'blue',
    '53': 'purple',
    '54': 'green',
}

# Function to process and plot for each sample
def process_and_plot(df_sample, sample_name, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    # Sort and filter the dataframe
    df_plot = df_sample.copy()
    df_plot = df_plot.sort_values(by='Lipid', key=lambda x: x.map(lipid_sort_key))
    df_plot = df_plot[~df_plot['Lipid'].str.contains(":0")]

    # Plot the ratios with the plot_ratios function
    plot_ratio(df_plot, color_mapping, output_directory, ratio_threshold=0.5)


### 
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_n9_n7_ratios(
    sample_csv_path, 
    manual_csv_path, 
    sample_label='CLAW Area', 
    manual_label='Manual Area', 
    save_dir='canola/plots/ratio/',
    filename_base='n9_n7_ratio_comparison',
    file_path=None
):
    """
    Create a scatter + line plot comparing n-9/n-7 ratios from CLAW and manual integration,
    and save it as PNG, PDF, and CSVs at 600 dpi.
    """
    # Load data
    df_sample = pd.read_csv(sample_csv_path)
    df_manual = pd.read_csv(manual_csv_path)

    df_sample['Lipid'] = df_sample['Lipid'].astype(str)
    df_manual['Lipid'] = df_manual['Lipid'].astype(str)

    # Remove square brackets from lipid names
    df_sample['Lipid'] = df_sample['Lipid'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    df_manual['Lipid'] = df_manual['Lipid'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)

    df_sample = df_sample.sort_values('Lipid')
    df_manual = df_manual.sort_values('Lipid')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sample['Lipid'], df_sample['n-9/n-7_Ratio'], '-o', color='red', label=sample_label, markersize=14)
    ax.plot(df_manual['Lipid'], df_manual['n-9/n-7_Ratio'], '-o', color='blue', label=manual_label, markersize=14)

    ax.set_xlabel("Lipid", fontsize=20)
    ax.set_ylabel("n-9 / n-7 Ratio", fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.legend(fontsize=16)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Setup file paths
    today_str = datetime.now().strftime("%Y%m%d")

    if file_path:
        base_path = file_path
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
    else:
        os.makedirs(save_dir, exist_ok=True)
        base_path = os.path.join(save_dir, filename_base)

    png_path = f"{base_path}.png"
    pdf_path = f"{base_path}.pdf"
    sample_csv_out = f"{base_path}_sample_df_{today_str}.csv"
    manual_csv_out = f"{base_path}_manual_df_{today_str}.csv"

    # Save plots
    plt.savefig(png_path, dpi=600)
    plt.savefig(pdf_path, dpi=600)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    # Save DataFrames
    df_sample.to_csv(sample_csv_out, index=False)
    df_manual.to_csv(manual_csv_out, index=False)
    print(f"Saved sample CSV: {sample_csv_out}")
    print(f"Saved manual CSV: {manual_csv_out}")

    plt.close(fig)


# ============================================================
# 🧬 Helper: split full dataframe into per-sample cleaned subsets
# ============================================================

import re
import pandas as pd
import os

def prepare_sample_ratio_dfs(df_with_ratio, save_dir=None, timestamp=None):
    """
    Splits the ratio dataframe by Sample_ID, cleans lipid names,
    removes unwanted TGs, sorts them, and optionally saves each as CSV.

    Parameters
    ----------
    df_with_ratio : pd.DataFrame
        Must contain ['Sample_ID', 'Lipid', 'Ratio'].
    save_dir : str, optional
        Directory to save cleaned CSVs.
    timestamp : str, optional
        Timestamp string to append to filenames (e.g. '11042025_9pm').

    Returns
    -------
    dict
        {'Crude': df1, 'Degummed': df2, 'RBD': df3, ...}
    """

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    # --- internal cleaning helpers ---
    def extract_tg(lipid_str):
        match = re.search(r'TG\(\d+:\d+\)', str(lipid_str))
        return match.group(0) if match else str(lipid_str)

    def process_dataframe(df):
        df = df.copy()
        df['Lipid'] = df['Lipid'].apply(extract_tg)
        df = df[~df['Lipid'].isin(['TG(53:0)', 'TG(54:6)', 'TG(52:5)'])]
        df = df.sort_values(by='Lipid').reset_index(drop=True)
        return df

    # --- group by sample and clean ---
    out_dfs = {}
    for sample_id in df_with_ratio['Sample_ID'].unique():
        df_sample = (
            df_with_ratio[df_with_ratio['Sample_ID'] == sample_id]
            .copy()
            .reset_index(drop=True)
        )
        df_sample_ratio = df_sample[df_sample['Ratio'].notna()].reset_index(drop=True)
        df_sample_ratio = process_dataframe(df_sample_ratio)
        out_dfs[sample_id] = df_sample_ratio

        if save_dir:
            ts = f"_{timestamp}" if timestamp else ""
            csv_path = os.path.join(save_dir, f"df_{sample_id}_ratio{ts}.csv")
            df_sample_ratio.to_csv(csv_path, index=False)
            print(f"✅ Saved: {csv_path}")

    return out_dfs


import pandas as pd
import os
from pathlib import Path

import pandas as pd
import os
from pathlib import Path

def calculate_ratio_statistics(
    data_dir=None,
    manual_dir=None,
    claw_files=None,
    manual_files=None,
    output_dir=None,
    save_file=True,
    ratio_column="n-9/n-7_Ratio",
    group_by_column="Lipid"
):
    """
    Calculate statistics for lipid ratios from CLAW and manual data files.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing the CLAW CSV files
    manual_dir : str or Path, optional
        Directory containing the manual CSV files. If None, uses data_dir
    claw_files : dict
        Dictionary mapping CLAW filenames to sample types
    manual_files : dict
        Dictionary mapping manual filenames to sample types
    output_dir : str or Path, optional
        Directory to save output file. If None, won't save.
    save_file : bool, default=True
        Whether to save the output CSV
    ratio_column : str, default="n-9/n-7_Ratio"
        Column name containing ratio values
    group_by_column : str, default="Lipid"
        Column to group by for statistics
    
    Returns:
    --------
    tuple : (df_CLAW, df_manual, std_ratios)
        Combined CLAW dataframe, combined manual dataframe, and statistics dataframe
    """
    data_dir = Path(data_dir)
    manual_dir = Path(manual_dir) if manual_dir else data_dir
    
    # Load and tag each CLAW dataframe
    print("Loading CLAW data...")
    dfs_claw = []
    for filename, sample_type in claw_files.items():
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df["Sample_Type"] = sample_type
        dfs_claw.append(df)
        print(f"  ✓ Loaded {filename}: {df.shape[0]} rows")
    
    # Combine into a single DataFrame
    df_CLAW = pd.concat(dfs_claw, ignore_index=True)
    print(f"Combined CLAW DataFrame: {df_CLAW.shape}")
    
    # Load and tag each Manual dataframe
    print("\nLoading Manual data...")
    dfs_manual = []
    for filename, sample_type in manual_files.items():
        file_path = manual_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df["Sample_Type"] = sample_type
        dfs_manual.append(df)
        print(f"  ✓ Loaded {filename}: {df.shape[0]} rows")
    
    # Combine into a single DataFrame
    df_manual = pd.concat(dfs_manual, ignore_index=True)
    print(f"Combined Manual DataFrame: {df_manual.shape}")
    
    # Calculate statistics
    print(f"\nCalculating statistics...")
    if ratio_column not in df_CLAW.columns:
        raise KeyError(f"Column '{ratio_column}' not found in CLAW data. Available columns: {df_CLAW.columns.tolist()}")
    
    std_ratios = (
        df_CLAW
        .groupby(group_by_column)[ratio_column]
        .std()
        .reset_index()
        .rename(columns={ratio_column: "Ratio_StdDev"})
    )
    
    print(f"\nStatistics calculated for {len(std_ratios)} unique {group_by_column}s")
    print("\nStandard deviation of n-9/n-7 Ratio:")
    print(std_ratios.head(10))
    if len(std_ratios) > 10:
        print(f"... and {len(std_ratios) - 10} more rows")
    
    # Save the file if requested
    if save_file and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = output_dir / f"ratio_stdev_by_{group_by_column.lower()}_{timestamp}.csv"
        
        std_ratios.to_csv(output_path, index=False)
        print(f"\n✓ File saved to: {output_path}")
    
    return df_CLAW, df_manual, std_ratios


import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

def load_rt_mapping(csv_path=None, default_map=None):
    """
    Load retention time mapping from a CSV file or use default dictionary.
    
    Parameters:
    -----------
    csv_path : str or Path, optional
        Path to CSV file with lipid-RT mappings (columns: Lipid, RT)
        If None, uses default_map
    default_map : dict, optional
        Default dictionary to use if csv_path is None
    
    Returns:
    --------
    dict
        Dictionary mapping lipid names to retention times
    """
    if csv_path is not None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"RT mapping file not found: {csv_path}")
        
        df_rt = pd.read_csv(csv_path)
        
        # Validate columns
        if 'Lipid' not in df_rt.columns or 'RT' not in df_rt.columns:
            raise ValueError(f"CSV must have 'Lipid' and 'RT' columns. Found: {df_rt.columns.tolist()}")
        
        rt_map = dict(zip(df_rt['Lipid'], df_rt['RT']))
        print(f"✓ Loaded RT mapping for {len(rt_map)} lipids from {csv_path.name}")
        
        return rt_map
    
    elif default_map is not None:
        print(f"✓ Using provided RT mapping for {len(default_map)} lipids")
        return default_map
    
    else:
        raise ValueError("Either csv_path or default_map must be provided")


def create_ratio_pivot_table(
    df,
    lipid_rt_map=None,
    rt_csv_path=None,
    ratio_column="n-9/n-7_Ratio",
    sample_column="Sample_Type",
    lipid_column="Lipid",
    sample_order=None,
    output_path=None,
    decimal_places=2
):
    """
    Create a pivot table of lipid ratios across different sample types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing lipid data
    lipid_rt_map : dict, optional
        Dictionary mapping lipid names to retention times (RT)
        Example: {"TG(52:2)_FA18:1": 18.05}
    rt_csv_path : str or Path, optional
        Path to CSV file with RT mappings (columns: Lipid, RT)
        If provided, takes precedence over lipid_rt_map
    ratio_column : str, default="n-9/n-7_Ratio"
        Column name containing ratio values to pivot
    sample_column : str, default="Sample_Type"
        Column name containing sample types
    lipid_column : str, default="Lipid"
        Column name containing lipid identifiers
    sample_order : list, optional
        List specifying the order of sample columns
    output_path : str or Path, optional
        Path to save the output CSV
    decimal_places : int, default=2
        Number of decimal places for rounding
    
    Returns:
    --------
    pd.DataFrame
        Pivot table with lipids as rows and sample types as columns
    """
    # Make a copy to avoid modifying original
    df_work = df.copy()
    
    print("="*60)
    print("CREATING RATIO PIVOT TABLE")
    print("="*60)
    print(f"Input data: {df_work.shape[0]} rows × {df_work.shape[1]} columns")
    
    # Load RT mapping
    if rt_csv_path is not None:
        rt_map = load_rt_mapping(csv_path=rt_csv_path)
    elif lipid_rt_map is not None:
        rt_map = load_rt_mapping(default_map=lipid_rt_map)
    else:
        raise ValueError("Either rt_csv_path or lipid_rt_map must be provided")
    
    # Add RT column
    df_work["RT (min)"] = df_work[lipid_column].map(rt_map)
    
    # Check for missing RTs
    missing_rt = df_work[df_work["RT (min)"].isna()][lipid_column].unique()
    if len(missing_rt) > 0:
        print(f"\n⚠️  Warning: {len(missing_rt)} lipids missing retention time:")
        for lipid in missing_rt[:5]:
            print(f"    • {lipid}")
        if len(missing_rt) > 5:
            print(f"    ... and {len(missing_rt) - 5} more")
    
    # Get mapped lipids
    mapped_lipids = df_work[df_work["RT (min)"].notna()]
    print(f"\nLipids with RT: {mapped_lipids[lipid_column].nunique()}")
    print(f"Sample types: {df_work[sample_column].unique().tolist()}")
    
    # Pivot to wide format
    df_pivot = df_work.pivot(
        index=[lipid_column, "RT (min)"], 
        columns=sample_column, 
        values=ratio_column
    ).reset_index()
    
    # Determine column order
    if sample_order is None:
        sample_columns = sorted([col for col in df_pivot.columns if col not in [lipid_column, "RT (min)"]])
    else:
        sample_columns = [col for col in sample_order if col in df_pivot.columns]
    
    # Reorder columns
    column_order = [lipid_column, "RT (min)"] + sample_columns
    df_pivot = df_pivot[column_order]
    
    # Round numeric columns
    numeric_columns = df_pivot.select_dtypes(include=['float64', 'float32']).columns
    df_pivot[numeric_columns] = df_pivot[numeric_columns].round(decimal_places)
    
    print(f"\nPivot table created: {df_pivot.shape[0]} rows × {df_pivot.shape[1]} columns")
    print(f"Columns: {', '.join(df_pivot.columns)}")
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pivot.to_csv(output_path, index=False)
        print(f"\n✓ Table saved to: {output_path}")
    
    return df_pivot





def calculate_ratio_difference_stats(
    df_claw,
    df_manual,
    claw_ratio_column="n-9/n-7_Ratio",
    manual_ratio_column="n-9/n-7_Ratio",
    lipid_column="Lipid",
    sample_column="Sample_Type",
    output_path=None,
    decimal_places=4
):
    """
    Calculate the standard deviation of differences between CLAW and Manual ratios.
    
    This function compares automated (CLAW) measurements with manual measurements
    to assess the consistency and accuracy of the automated method.
    
    Parameters:
    -----------
    df_claw : pd.DataFrame
        Dataframe containing CLAW (automated) ratio data
    df_manual : pd.DataFrame
        Dataframe containing manual ratio data
    claw_ratio_column : str, default="n-9/n-7_Ratio"
        Column name for CLAW ratio values
    manual_ratio_column : str, default="n-9/n-7_Ratio"
        Column name for manual ratio values
    lipid_column : str, default="Lipid"
        Column name for lipid identifiers
    sample_column : str, default="Sample_Type"
        Column name for sample types
    output_path : str or Path, optional
        Path to save the output CSV
    decimal_places : int, default=4
        Number of decimal places for rounding
    
    Returns:
    --------
    tuple : (merged_df, std_by_sample)
        - merged_df: DataFrame with CLAW, Manual, and Difference columns
        - std_by_sample: DataFrame with standard deviation by sample type
    """
    print("="*60)
    print("CALCULATING RATIO DIFFERENCE STATISTICS")
    print("="*60)
    
    # Step 1: Rename columns for clarity
    df_claw_renamed = df_claw.rename(columns={claw_ratio_column: "CLAW_Ratio"})
    df_manual_renamed = df_manual.rename(columns={manual_ratio_column: "Manual_Ratio"})
    
    print(f"CLAW data: {df_claw_renamed.shape[0]} rows")
    print(f"Manual data: {df_manual_renamed.shape[0]} rows")
    
    # Step 2: Merge on Lipid + Sample_Type
    merge_keys = [lipid_column, sample_column]
    merged = pd.merge(
        df_claw_renamed[[lipid_column, sample_column, "CLAW_Ratio"]],
        df_manual_renamed[[lipid_column, sample_column, "Manual_Ratio"]],
        on=merge_keys,
        how="inner"
    )
    
    print(f"\nMatched pairs: {len(merged)}")
    
    # Check for missing matches
    claw_pairs = set(zip(df_claw_renamed[lipid_column], df_claw_renamed[sample_column]))
    manual_pairs = set(zip(df_manual_renamed[lipid_column], df_manual_renamed[sample_column]))
    
    claw_only = claw_pairs - manual_pairs
    manual_only = manual_pairs - claw_pairs
    
    if claw_only:
        print(f"⚠️  {len(claw_only)} lipid-sample pairs only in CLAW data")
    if manual_only:
        print(f"⚠️  {len(manual_only)} lipid-sample pairs only in Manual data")
    
    # Step 3: Calculate difference between ratios
    merged["Difference"] = merged["CLAW_Ratio"] - merged["Manual_Ratio"]
    merged["Absolute_Difference"] = merged["Difference"].abs()
    merged["Percent_Difference"] = ((merged["Difference"] / merged["Manual_Ratio"]) * 100).round(2)
    
    print(f"\nOverall statistics:")
    print(f"  Mean difference: {merged['Difference'].mean():.4f}")
    print(f"  Mean absolute difference: {merged['Absolute_Difference'].mean():.4f}")
    print(f"  Median difference: {merged['Difference'].median():.4f}")
    print(f"  Overall std dev: {merged['Difference'].std():.4f}")
    
    # Step 4: Group by Sample_Type and calculate std dev
    std_by_sample = (
        merged.groupby(sample_column)["Difference"]
        .agg(['std', 'mean', 'count'])
        .reset_index()
        .rename(columns={
            'std': 'Difference_StdDev',
            'mean': 'Mean_Difference',
            'count': 'N_Pairs'
        })
    )
    
    # Round results
    numeric_cols = ['Difference_StdDev', 'Mean_Difference']
    std_by_sample[numeric_cols] = std_by_sample[numeric_cols].round(decimal_places)
    
    print(f"\nResults by sample type:")
    for _, row in std_by_sample.iterrows():
        print(f"  {row[sample_column]}:")
        print(f"    • N = {row['N_Pairs']}")
        print(f"    • Mean diff = {row['Mean_Difference']:.4f}")
        print(f"    • Std dev = {row['Difference_StdDev']:.4f}")
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        std_by_sample.to_csv(output_path, index=False)
        print(f"\n✓ Statistics saved to: {output_path}")
        
        # Also save detailed merged data
        detailed_path = output_path.parent / output_path.name.replace('.csv', '_detailed.csv')
        merged.to_csv(detailed_path, index=False)
        print(f"✓ Detailed comparison saved to: {detailed_path}")
    
    return merged, std_by_sample


def plot_n9_n7_ratios_intensity(
    sample_csv_path, 
    manual_csv_path, 
    sample_label='CLAW Intensity', 
    manual_label='Manual Area', 
    save_dir='canola/plots/ratio/',
    filename_base='n9_n7_intensity_ratio_comparison',
    file_path=None
):
    """
    Create a scatter + line plot comparing n-9/n-7 INTENSITY ratios from CLAW 
    with manual AREA integration ratios, and save it as PNG, PDF, and CSVs at 600 dpi.
    
    Parameters
    ----------
    sample_csv_path : str
        Path to CSV containing CLAW data with 'n-9/n-7_Intensity_Ratio' column
    manual_csv_path : str
        Path to CSV containing manual data with 'n-9/n-7_Ratio' column
    sample_label : str, default='CLAW Intensity'
        Label for the CLAW intensity data in the plot legend
    manual_label : str, default='Manual Area'
        Label for the manual area data in the plot legend
    save_dir : str, default='canola/plots/ratio/'
        Directory to save output files (used if file_path is None)
    filename_base : str, default='n9_n7_intensity_ratio_comparison'
        Base filename for output files (used if file_path is None)
    file_path : str, optional
        Complete file path (without extension) for output files.
        If provided, overrides save_dir and filename_base
    
    Returns
    -------
    None
        Saves PNG, PDF, and CSV files to disk
    
    Notes
    -----
    - Sample data uses 'n-9/n-7_Intensity_Ratio' column
    - Manual data uses 'n-9/n-7_Ratio' column
    - Plots are saved at 600 dpi resolution
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    
    # Load data
    df_sample = pd.read_csv(sample_csv_path)
    df_manual = pd.read_csv(manual_csv_path)

    # Convert Lipid to string and sort
    df_sample['Lipid'] = df_sample['Lipid'].astype(str)
    df_manual['Lipid'] = df_manual['Lipid'].astype(str)

    # Remove square brackets from lipid names
    df_sample['Lipid'] = df_sample['Lipid'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    df_manual['Lipid'] = df_manual['Lipid'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)

    df_sample = df_sample.sort_values('Lipid')
    df_manual = df_manual.sort_values('Lipid')

    # Validate required columns
    if 'n-9/n-7_Intensity_Ratio' not in df_sample.columns:
        raise KeyError(f"Sample CSV missing 'n-9/n-7_Intensity_Ratio' column. Available columns: {df_sample.columns.tolist()}")
    if 'n-9/n-7_Ratio' not in df_manual.columns:
        raise KeyError(f"Manual CSV missing 'n-9/n-7_Ratio' column. Available columns: {df_manual.columns.tolist()}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sample['Lipid'], df_sample['n-9/n-7_Intensity_Ratio'], '-o', 
            color='red', label=sample_label, markersize=14)
    ax.plot(df_manual['Lipid'], df_manual['n-9/n-7_Ratio'], '-o', 
            color='blue', label=manual_label, markersize=14)

    ax.set_xlabel("Lipid", fontsize=20)
    ax.set_ylabel("n-9 / n-7 Ratio", fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.legend(fontsize=16)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Setup file paths
    today_str = datetime.now().strftime("%Y%m%d")

    if file_path:
        base_path = file_path
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
    else:
        os.makedirs(save_dir, exist_ok=True)
        base_path = os.path.join(save_dir, filename_base)

    png_path = f"{base_path}.png"
    pdf_path = f"{base_path}.pdf"
    sample_csv_out = f"{base_path}_sample_intensity_df_{today_str}.csv"
    manual_csv_out = f"{base_path}_manual_df_{today_str}.csv"

    # Save plots
    plt.savefig(png_path, dpi=600)
    plt.savefig(pdf_path, dpi=600)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    # Save DataFrames
    df_sample.to_csv(sample_csv_out, index=False)
    df_manual.to_csv(manual_csv_out, index=False)
    print(f"Saved sample intensity CSV: {sample_csv_out}")
    print(f"Saved manual CSV: {manual_csv_out}")

    plt.close(fig)