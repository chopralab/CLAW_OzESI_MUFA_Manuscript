import os
import pandas as pd
import numpy as np
import pymzml
from collections import defaultdict


# CLAW_OzESI.py
from scipy.signal import find_peaks, peak_widths

# ========= Helper Functions (unchanged) =========

def create_ion_dict(mrm_database):
    ion_dict = defaultdict(list)
    for index, row in mrm_database.iterrows():
        ion_dict[(row['Parent_Ion'], row['Product_Ion'])].append((row['Lipid'], row['Class']))
    return ion_dict

def within_tolerance(a, b, tolerance=0.3):
    return abs(a - b) <= tolerance

def match_ions(row, ion_dict, tolerance=0.3):
    ions = (row['Parent_Ion'], row['Product_Ion'])
    matched_lipids = []
    matched_classes = []
    for key, value in ion_dict.items():
        if within_tolerance(ions[0], key[0], tolerance) and within_tolerance(ions[1], key[1], tolerance):
            matched_lipids.extend([match[0] for match in value])
            matched_classes.extend([match[1] for match in value])
    if matched_lipids and matched_classes:
        row['Lipid'] = ' | '.join(matched_lipids)
        row['Class'] = ' | '.join(matched_classes)
    return row

def mzml_parser(file_path, plot_chromatogram=False):
    # Note: This function updates global DataFrames.
    global master_df, OzESI_time_df
    rows = []
    ozesi_rows = []
    
    run = pymzml.run.Reader(file_path, skip_chromatogram=False)
    q1_mz = 0
    q3_mz = 0

    for spectrum in run:
        for element in spectrum.ID.split(' '):
            if 'Q1' in element:
                q1 = element.split('=')
                q1_mz = float(q1[1])
            if 'Q3' in element:
                q3 = element.split('=')
                q3_mz = float(q3[1])
                intensity_store = np.array([intensity for _, intensity in spectrum.peaks()])
                intensity_sum = np.sum(intensity_store)
                transition = f"{q1_mz} -> {q3_mz}"
                sample_id = os.path.basename(file_path).replace('.mzML', '')
                rows.append({
                    'Parent_Ion': q1_mz,
                    'Product_Ion': q3_mz,
                    'Intensity': intensity_sum,
                    'Transition': transition,
                    'Sample_ID': sample_id
                })
                for time_val, intensity in spectrum.peaks():
                    ozesi_rows.append({
                        'Parent_Ion': q1_mz,
                        'Product_Ion': q3_mz,
                        'Retention_Time': time_val,
                        'OzESI_Intensity': intensity,
                        'Sample_ID': sample_id,
                        'Transition': transition
                    })
    df = pd.DataFrame(rows)
    OzESI_time_df = pd.concat([OzESI_time_df, pd.DataFrame(ozesi_rows)], ignore_index=True)
    master_df = pd.concat([master_df, df], ignore_index=True)
    print(f'Finished parsing mzML file: {file_path}\n')

def mzml_parser_batch(folder_name, plot_chromatogram=False):
    global master_df
    data_folder = os.listdir(folder_name)
    data_folder.sort()
    for file in data_folder:
        if file.endswith('.mzML'):
            file_path = os.path.join(folder_name, file)
            mzml_parser(file_path, plot_chromatogram=plot_chromatogram)
    print('Finished parsing all mzML files\n')

def save_dataframe(df, Project_results, file_name_to_save, max_attempts=5):
    folder_path = f'data_results/data/data_matching/{Project_results}'
    os.makedirs(folder_path, exist_ok=True)
    for i in range(max_attempts):
        file_path = f'{folder_path}/{file_name_to_save}.csv'
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
            print(f"Saved DataFrame to {file_path}")
            break
    else:
        print(f"Failed to save DataFrame after {max_attempts} attempts.")

def read_mrm_list(filename, remove_std=True, deuterated=False):
    raw_mrm_data = pd.read_excel(filename, sheet_name=None)
    concatenated_mrm_data = pd.concat(raw_mrm_data, ignore_index=True)
    lipid_MRM_data = concatenated_mrm_data[['Compound Name', 'Parent Ion', 'Product Ion', 'Class']]
    lipid_MRM_data.columns = lipid_MRM_data.columns.str.replace(' ', '_')
    lipid_MRM_data['Parent_Ion'] = np.round(lipid_MRM_data['Parent_Ion'], 1)
    lipid_MRM_data['Product_Ion'] = np.round(lipid_MRM_data['Product_Ion'], 1)
    lipid_MRM_data['Transition'] = lipid_MRM_data['Parent_Ion'].astype(str) + ' -> ' + lipid_MRM_data['Product_Ion'].astype(str)
    lipid_MRM_data = lipid_MRM_data.rename(columns={'Compound_Name': 'Lipid'})
    if remove_std:
        lipid_classes_to_keep = ['PS', 'PG', 'CE', 'PC', 'DAG', 'PE', 'TAG', 'FA', 'Cer', 'CAR', 'PI', 'SM']
        lipid_MRM_data = lipid_MRM_data[lipid_MRM_data['Class'].isin(lipid_classes_to_keep)]
    if deuterated:
        lipid_MRM_data['Parent_Ion'] += 1
        lipid_MRM_data['Product_Ion'] += 1
        lipid_MRM_data['Transition'] = lipid_MRM_data['Parent_Ion'].astype(str) + ' -> ' + lipid_MRM_data['Product_Ion'].astype(str)
    return lipid_MRM_data

def match_lipids_parser(mrm_database, df, tolerance=0.3):
    ion_dict = create_ion_dict(mrm_database)
    df_matched = df.apply(lambda row: match_ions(row, ion_dict=ion_dict, tolerance=tolerance), axis=1)
    return df_matched

# ========= Global DataFrame Initialization =========

# These globals are used by the mzML parsing functions.
master_df = pd.DataFrame(columns=['Parent_Ion', 'Product_Ion', 'Intensity', 'Transition', 'Sample_ID'])
OzESI_time_df = pd.DataFrame(columns=['Lipid', 'Parent_Ion', 'Product_Ion', 'Retention_Time', 'OzESI_Intensity', 'Sample_ID', 'Transition'])

# ========= The Parse Class =========

class Parse:
    def __init__(self, data_base_name_location, Project_Folder_data, Project_results, 
                 file_name_to_save, tolerance, remove_std=True, save_data=False, 
                 batch_processing=True, plot_chromatogram=False):
        """
        Initializes a new Parse instance with the given configuration.
        """
        self.data_base_name_location = data_base_name_location
        self.Project_Folder_data = Project_Folder_data
        self.Project_results = Project_results
        self.file_name_to_save = file_name_to_save
        self.tolerance = tolerance
        self.remove_std = remove_std
        self.save_data = save_data
        self.batch_processing = batch_processing
        self.plot_chromatogram = plot_chromatogram

    def read_mrm_list(self, deuterated=False):
        """
        Reads the MRM list (Excel file) using the stored database location.
        """
        # Use the instance’s configuration for filename and removal flag.
        mrm_database = read_mrm_list(self.data_base_name_location, remove_std=self.remove_std, deuterated=deuterated)
        return mrm_database

    def match_lipids_parser(self, mrm_database, df):
        """
        Matches lipids from the parsed data using the given MRM database.
        """
        # Uses the global function match_lipids_parser with self.tolerance.
        return match_lipids_parser(mrm_database, df, tolerance=self.tolerance)

    def full_parse(self):
        """
        Runs the full parsing workflow:
          1. Resets the global DataFrames.
          2. Reads the MRM database.
          3. Parses mzML files (either a batch or a single file).
          4. Matches the parsed data against the MRM list.
          5. Optionally saves the results.
          
        Returns:
            tuple: (df_matched, OzESI_time_df)
        """
        global master_df, OzESI_time_df
        # (Optional) Reset the global DataFrames before processing
        master_df = pd.DataFrame(columns=['Parent_Ion', 'Product_Ion', 'Intensity', 'Transition', 'Sample_ID'])
        OzESI_time_df = pd.DataFrame(columns=['Lipid', 'Parent_Ion', 'Product_Ion', 'Retention_Time', 
                                               'OzESI_Intensity', 'Sample_ID', 'Transition'])
        
        # Step 1: Read the MRM database.
        mrm_database = self.read_mrm_list()
        
        # Step 2: Parse mzML files.
        if self.batch_processing:
            mzml_parser_batch(self.Project_Folder_data, plot_chromatogram=self.plot_chromatogram)
        else:
            mzml_parser(self.Project_Folder_data, plot_chromatogram=self.plot_chromatogram)
        
        # Step 3: Match lipids using the parsed master_df.
        df_matched = self.match_lipids_parser(mrm_database, master_df)
        
        # Step 4: Optionally save the matched DataFrame.
        if self.save_data:
            save_dataframe(df_matched, self.Project_results, self.file_name_to_save)
        
        return df_matched, OzESI_time_df

    def mrm_run_all(self, deuterated=False):
        """
        Runs the complete workflow in one function:
          1) Runs full_parse to process the mzML files and perform an initial match.
          2) Re-reads the MRM list (with an option to adjust for deuterated compounds).
          3) Matches the parsed OzESI chromatogram data against the MRM list.
          
        Returns:
            tuple: (df_full_matched, OzESI_time_df, df_OzESI_matched)
                - df_full_matched: The matched DataFrame from full_parse (based on master_df).
                - OzESI_time_df: The DataFrame containing OzESI chromatogram data.
                - df_OzESI_matched: The OzESI DataFrame with lipid matching using the (optionally re-read) MRM database.
        """
        # Step 1: Run the full parse workflow.
        df_full_matched, ozesi_time_df = self.full_parse()
        
        # Step 2: Re-read the MRM database (with the provided deuterated flag).
        mrm_database = self.read_mrm_list(deuterated=deuterated)
        
        # Step 3: Match the OzESI chromatogram data using the re-read MRM database.
        df_OzESI_matched = self.match_lipids_parser(mrm_database, ozesi_time_df)

        
        return df_full_matched, ozesi_time_df, df_OzESI_matched




def process_peaks(df, rel_height=0.95, **find_peaks_kwargs):
    """
    Process chromatographic peaks grouped by Parent_Ion and Sample_ID.
    Detects peaks, integrates them, and adds peak boundary and area columns.
    """

    def process_peaks_for_group(group, rel_height=rel_height, **find_peaks_kwargs):
        group = group.sort_values('Retention_Time').copy()
        rt = group['Retention_Time'].values
        intensity = group['OzESI_Intensity'].values

        peaks, properties = find_peaks(intensity, **find_peaks_kwargs)
        widths, width_heights, left_ips, right_ips = peak_widths(intensity, peaks, rel_height=rel_height)

        group['peak_area'] = np.nan
        group['peak_start'] = np.nan
        group['peak_stop'] = np.nan
        group['peak_middle'] = np.nan

        for i, peak_idx in enumerate(peaks):
            left_rt = np.interp(left_ips[i], np.arange(len(rt)), rt)
            right_rt = np.interp(right_ips[i], np.arange(len(rt)), rt)
            peak_rt = np.interp(peak_idx, np.arange(len(rt)), rt)

            # Constrain to ±0.6 from center
            if (peak_rt - left_rt) > 0.6:
                left_rt = peak_rt - 0.6
            if (right_rt - peak_rt) > 0.6:
                right_rt = peak_rt + 0.6

            left_index = np.searchsorted(rt, left_rt, side='left')
            right_index = np.searchsorted(rt, right_rt, side='right') - 1
            left_index = max(left_index, 0)
            right_index = min(right_index, len(rt) - 1)

            rt_segment = rt[left_index:right_index+1]
            intensity_segment = intensity[left_index:right_index+1]
            area = np.trapz(intensity_segment, rt_segment)

            mask = (group['Retention_Time'] >= left_rt) & (group['Retention_Time'] <= right_rt)
            group.loc[mask, 'peak_area'] = area
            group.loc[mask, 'peak_start'] = left_rt
            group.loc[mask, 'peak_stop'] = right_rt
            group.loc[mask, 'peak_middle'] = peak_rt

        return group

    processed_df = df.groupby(['Parent_Ion', 'Sample_ID'], group_keys=False).apply(
        process_peaks_for_group, rel_height=rel_height, **find_peaks_kwargs
    )

    return processed_df


def ratio_peak_area(df_CrudeCanola):
    """
    Computes n-9/n-7 ratio of peak areas for each lipid.
    Adds a new column 'Ratio_Area' to the DataFrame.

    Parameters
    ----------
    df_CrudeCanola : pandas.DataFrame
        Must contain columns ['Lipid', 'db_pos', 'peak_area']

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with new column 'Ratio_Area'
    """
    df = df_CrudeCanola.copy()
    df['Ratio_Area'] = np.nan

    for lipid, group in df.groupby('Lipid'):
        if {'n-9', 'n-7'}.issubset(set(group['db_pos'])):
            peak_area_n9 = group.loc[group['db_pos'] == 'n-9', 'peak_area']
            peak_area_n7 = group.loc[group['db_pos'] == 'n-7', 'peak_area']

            if not peak_area_n9.empty and not peak_area_n7.empty:
                ratio = peak_area_n9.iloc[0] / peak_area_n7.iloc[0]
                idx = group.loc[group['db_pos'] == 'n-9'].index
                df.loc[idx, 'Ratio_Area'] = ratio

    return df


def ion_label_parser(lipid_mrm, df_OzESI_matched, ion_tolerance=0.3, rt_tolerance=1.0, ion_labels=['n-7', 'n-9']):
    """
    Parse and match OzESI data against lipid MRM database with ion labels (e.g., n-7, n-9).
    
    This function identifies specific lipid isomers by matching precursor ions, product ions,
    and retention times within specified tolerance windows.
    
    Parameters
    ----------
    lipid_mrm : pandas.DataFrame
        DataFrame containing lipid MRM reference data with columns:
        - 'Lipid': Lipid name
        - 'Product_Ion': Product ion m/z
        - 'Retention_Time': Expected retention time
        - Column names matching ion_labels (e.g., 'n-7', 'n-9'): Precursor ion m/z values
    
    df_OzESI_matched : pandas.DataFrame
        DataFrame from OzESI parsing containing columns:
        - 'Parent_Ion': Precursor ion m/z
        - 'Product_Ion': Product ion m/z
        - 'Retention_Time': Observed retention time
        - 'OzESI_Intensity': Signal intensity
        - 'Sample_ID': Sample identifier
        - 'Transition': Ion transition string
    
    ion_tolerance : float, optional
        Tolerance window for ion m/z matching in Daltons (default: 0.3)
    
    rt_tolerance : float, optional
        Tolerance window for retention time matching in minutes (default: 1.0)
    
    ion_labels : list of str, optional
        List of ion label column names to process (default: ['n-7', 'n-9'])
    
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with columns:
        - 'Lipid': Assigned lipid name
        - 'db_pos': Double bond position label (e.g., 'n-7', 'n-9')
        - 'Parent_Ion': Matched parent ion m/z
        - 'Product_Ion': Matched product ion m/z
        - 'Retention_Time': Observed retention time
        - 'OzESI_Intensity': Signal intensity
        - 'Sample_ID': Sample identifier
        - 'Transition': Ion transition string
    
    Examples
    --------
    >>> lipid_mrm = pd.read_csv('Projects/test_area/lipid_mrm.csv')
    >>> df_canola_lipids = ion_label_parser(lipid_mrm, df_OzESI_matched, 
    ...                                      ion_tolerance=0.3, rt_tolerance=1.0)
    """
    results = []
    
    # Iterate over rows in lipid_mrm
    for _, row in lipid_mrm.iterrows():
        lipid_name = row['Lipid']
        product_ion = row['Product_Ion']
        expected_rt = row['Retention_Time']
        
        for ion_label in ion_labels:
            # Skip if ion_label column doesn't exist or value is NaN
            if ion_label not in row.index or pd.isna(row[ion_label]):
                continue
                
            precursor_ion = row[ion_label]
            
            # Apply filtering conditions
            matches = df_OzESI_matched[
                (df_OzESI_matched['Parent_Ion'].between(precursor_ion - ion_tolerance, 
                                                         precursor_ion + ion_tolerance)) & 
                (df_OzESI_matched['Product_Ion'].between(product_ion - ion_tolerance, 
                                                          product_ion + ion_tolerance)) & 
                (df_OzESI_matched['Retention_Time'].between(expected_rt - rt_tolerance, 
                                                             expected_rt + rt_tolerance))
            ].copy()
            
            # Assign lipid info
            matches['Lipid'] = lipid_name
            matches['db_pos'] = ion_label
            
            # Append to results
            results.append(matches)
    
    # Concatenate all results into final DataFrame
    if results:
        df_result = pd.concat(results, ignore_index=True)
        
        # Reorder columns neatly
        df_result = df_result[[
            'Lipid', 'db_pos', 'Parent_Ion', 'Product_Ion', 'Retention_Time',
            'OzESI_Intensity', 'Sample_ID', 'Transition'
        ]]
    else:
        # Return empty DataFrame with expected columns if no matches found
        df_result = pd.DataFrame(columns=[
            'Lipid', 'db_pos', 'Parent_Ion', 'Product_Ion', 'Retention_Time',
            'OzESI_Intensity', 'Sample_ID', 'Transition'
        ])
    
    return df_result

def peak_analysis(df_canola_lipids, output_dir, show_plots=False, plot_TG_module=None):
    """
    Analyze lipid peaks across multiple samples with sample-aware configuration.
    
    This function processes each sample in the dataset, identifies TG patterns,
    analyzes peak areas and intensities for n-7 and n-9 isomers, calculates ratios, 
    and saves summary results to CSV files.
    
    Parameters
    ----------
    df_canola_lipids : pandas.DataFrame
        DataFrame containing matched lipid data with columns:
        - 'Sample_ID': Sample identifier
        - 'Lipid': Lipid name (should contain TG patterns like [TG(52:3)])
        - 'db_pos': Double bond position (e.g., 'n-7', 'n-9')
        - 'OzESI_Intensity': Intensity values for peak analysis
        - Other columns required by peak analysis functions
    
    output_dir : str
        Directory path where summary CSV files will be saved
    
    show_plots : bool, optional
        If True, display plots during analysis. If False, suppress plots (default: False)
    
    plot_TG_module : module, optional
        The plot_TG module containing analysis functions. If None, will attempt
        to import plot_TG from the current environment (default: None)
    
    Returns
    -------
    pandas.DataFrame
        Combined summary DataFrame for all samples with columns:
        - 'Sample_ID': Sample identifier
        - 'Lipid': Lipid pattern (e.g., TG(52:3))
        - 'n-7_Area': Peak area for n-7 isomer
        - 'n-9_Area': Peak area for n-9 isomer
        - 'n-9/n-7_Ratio': Area-based ratio
        - 'n-7_Intensity': Peak intensity for n-7 isomer
        - 'n-9_Intensity': Peak intensity for n-9 isomer
        - 'n-9/n-7_Intensity_Ratio': Intensity-based ratio
    
    Side Effects
    ------------
    - Creates output_dir if it doesn't exist
    - Saves per-sample CSV files: df_{sample_id}_summary.csv
    - Saves combined CSV file: df_all_samples_summary.csv
    
    Examples
    --------
    >>> import plot_TG
    >>> output_path = "Projects/test_area/results"
    >>> summary_df = peak_analysis(df_canola_lipids, output_path, show_plots=False)
    >>> print(summary_df.head())
    """
    import os
    import re
    import copy
    import pandas as pd
    import numpy as np
    
    # Import plot_TG if not provided
    if plot_TG_module is None:
        try:
            import plot_TG as plot_TG_module
        except ImportError:
            raise ImportError("plot_TG module is required but not found. Please provide it as a parameter.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all unique samples
    sample_ids = df_canola_lipids["Sample_ID"].unique()
    print(f"Found {len(sample_ids)} samples:")
    for sid in sample_ids:
        print(" •", sid)
    
    master_summaries = []  # to aggregate across samples
    
    # Process each sample
    for sample_id in sample_ids:
        print(f"\n🔍 Processing sample: {sample_id}")
        
        # Per-sample configuration (fallback to generic if the sample-aware factory isn't present)
        if hasattr(plot_TG_module, "create_custom_config_for_sample"):
            custom_config = plot_TG_module.create_custom_config_for_sample(sample_id)
        else:
            custom_config = plot_TG_module.create_custom_config()
        
        # Safe filename for per-sample CSV
        safe_name = re.sub(r"\W+", "_", sample_id)
        save_df = f"df_{safe_name}_summary.csv"
        
        # Filter to this sample
        sample_df = df_canola_lipids[df_canola_lipids["Sample_ID"] == sample_id]
        
        # Extract TG patterns like TG(52:3), TG(54:4), etc. from Lipid labels
        # NEW - keeps _FA18:1
        lipid_patterns = sorted({
            m.group(0) for lipid in sample_df["Lipid"].unique()
            if (m := re.search(r"\[TG\(\d+:\d+\)\]_FA\d+:\d+", str(lipid)))
        })
        if not lipid_patterns:
            print(f"⚠️  No TG patterns found for sample '{sample_id}'. Skipping analysis.")
            continue
        
        # Analyze each lipid pattern for this sample
        all_results = {}
        for pattern in lipid_patterns:
            pattern_regex = re.escape(pattern)  # safer than manual paren escaping
            # If apply_* mutates the config, protect the base by copying
            lipid_config = plot_TG_module.apply_lipid_specific_config(copy.deepcopy(custom_config), pattern)
            result = plot_TG_module.analyze_lipid_peaks_with_peakfinder(
                df_canola_lipids, pattern_regex, sample_id, lipid_config, show_plots=show_plots
            )
            all_results[pattern] = result
        
        # Visualize ratios for this sample
        plot_TG_module.visualize_lipid_ratios(all_results, sample_id, show_plots=show_plots)
        
        # Build the per-sample summary table with both area and intensity metrics
        summary_rows = []
        for lipid, data in all_results.items():
            # Extract area-based metrics
            n7_area = data.get("n-7", {}).get("largest_peak_area", 0)
            n9_area = data.get("n-9", {}).get("largest_peak_area", 0)
            area_ratio = data.get("ratio", None)
            
            # Calculate intensity-based metrics
            # Filter data for this specific lipid pattern and sample
            lipid_mask = sample_df["Lipid"].str.contains(re.escape(lipid), na=False, regex=True)
            lipid_data = sample_df[lipid_mask]
            
            # Extract n-7 and n-9 intensity values
            if 'db_pos' in lipid_data.columns and 'OzESI_Intensity' in lipid_data.columns:
                n7_intensities = lipid_data[lipid_data['db_pos'] == 'n-7']['OzESI_Intensity']
                n9_intensities = lipid_data[lipid_data['db_pos'] == 'n-9']['OzESI_Intensity']
                
                # Use max intensity as the representative value (can be changed to sum or mean)
                n7_intensity = n7_intensities.max() if len(n7_intensities) > 0 else 0
                n9_intensity = n9_intensities.max() if len(n9_intensities) > 0 else 0
                
                # Calculate intensity ratio
                if n7_intensity > 0:
                    intensity_ratio = n9_intensity / n7_intensity
                else:
                    intensity_ratio = None
            else:
                n7_intensity = None
                n9_intensity = None
                intensity_ratio = None
                if 'OzESI_Intensity' not in lipid_data.columns:
                    print(f"  ⚠️  Warning: 'OzESI_Intensity' column not found for {lipid}")
                if 'db_pos' not in lipid_data.columns:
                    print(f"  ⚠️  Warning: 'db_pos' column not found for {lipid}")
            
            summary_rows.append({
                "Sample_ID": sample_id,
                "Lipid": lipid,
                "n-7_Area": n7_area,
                "n-9_Area": n9_area,
                "n-9/n-7_Ratio": area_ratio,
                "n-7_Intensity": n7_intensity,
                "n-9_Intensity": n9_intensity,
                "n-9/n-7_Intensity_Ratio": intensity_ratio
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save per-sample CSV
        save_path = os.path.join(output_dir, save_df)
        summary_df.to_csv(save_path, index=False)
        print(f"✅ Saved summary results for {sample_id} → {save_path}")
        print(f"   Columns: {', '.join(summary_df.columns)}")
        
        master_summaries.append(summary_df)
    
    # Save a combined master summary for all samples
    if master_summaries:
        master_df = pd.concat(master_summaries, ignore_index=True)
        master_path = os.path.join(output_dir, "df_all_samples_summary.csv")
        master_df.to_csv(master_path, index=False)
        print(f"\n📦 Saved combined summary for all samples → {master_path}")
        print(f"   Shape: {master_df.shape[0]} rows × {master_df.shape[1]} columns")
        print(f"   Columns: {', '.join(master_df.columns)}")
        return master_df
    else:
        print("\nℹ️ No summaries were generated (no patterns found for any samples).")
        return pd.DataFrame()