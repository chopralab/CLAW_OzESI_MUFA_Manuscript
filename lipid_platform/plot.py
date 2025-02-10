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

    # Show the plot
    fig.show()


#### Comparison Plot CLAW vs Manual

import pandas as pd
import matplotlib.pyplot as plt

# Manual Canola data for comparison
manual_canola = {
    'TG': ['TG(52:2)', 'TG(52:3)', 'TG(52:4)', 'TG(54:2)', 'TG(54:3)', 'TG(54:4)', 'TG(54:5)'],
    'Crude': [3.92, 2.2, 2.31, 4.27, 4.88, 3.78, 5.58],
    'Degummed': [3.76, 3.25, 2.25, 3.03, 4.94, 4.06, 5.65],
    'RBD': [4.13, 2.34, 2.18, 4.36, 4.57, 4.04, 4.45]
}

# Creating DataFrame for Manual Canola data
manual_df = pd.DataFrame(manual_canola)

# Function to plot CLAW vs Caitlin data for each sample DataFrame
def plot_ratio_comparison(df_sample_ratio, plot_title, caitlin_column):
    plt.figure(figsize=(10, 6))
    
    # Plotting sample data (Black line, circle markers)
    plt.plot(df_sample_ratio['Lipid'], df_sample_ratio['Ratio'], label='CLAW Ratio', color='black', marker='o', markersize=12, linewidth=2)
    
    # Plotting Manual data (Blue line, circle markers)
    plt.plot(manual_df['TG'], manual_df[caitlin_column], label=f"Manual {plot_title}", color='blue', marker='o', markersize=12, linewidth=2)
    
    # Adding Title, Labels, and Legend
    plt.title(f'{plot_title} n-9 n-7 Ratios', fontsize=14)
    plt.xlabel('Lipid', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.legend()
    
    # Adding grid, rotating x-axis labels, and showing the plot
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Saving the plot to a file
    plt.savefig(f'Projects/canola/plots/{plot_title} n-9 n-7 Ratios.png', dpi=300)
    
    # Show the plot
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
    if output_file:
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
