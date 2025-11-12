"""
Script: CLAW_OzESI.py

Description:
    This script processes mzML files to extract ion transitions and intensities for lipid analysis,
    and matches these transitions against a provided MRM (Multiple Reaction Monitoring) database.
    It handles both a "master" set of parsed data as well as OzESI chromatogram data.
    
Inputs:
    - mzML files: Mass spectrometry files containing spectra with ion transitions (Parent_Ion and Product_Ion)
      and retention times.
    - MRM database: An Excel file containing target lipids, their parent and product ion values, and their classes.
    
Outputs:
    - master_df: A DataFrame containing parsed data with columns: 
          ['Parent_Ion', 'Product_Ion', 'Intensity', 'Transition', 'Sample_ID'].
    - OzESI_time_df: A DataFrame containing chromatogram data with columns:
          ['Lipid', 'Parent_Ion', 'Product_Ion', 'Retention_Time', 'OzESI_Intensity', 'Sample_ID', 'Transition'].
    - Optionally, the script can save the matched DataFrame as a CSV file in a specified results folder.
    
Usage:
    The script defines a Parse class that encapsulates the full workflow:
        1. Reading the MRM database.
        2. Parsing one or more mzML files.
        3. Matching parsed data against the MRM database.
        4. Optionally saving the results.
    
    Instantiate the Parse class with the desired configuration and call its methods (e.g., full_parse or mrm_run_all)
    to execute the analysis.
    
Author: Your Name (or Organization)
Date: YYYY-MM-DD
"""

import os
import pandas as pd
import numpy as np
import pymzml
from collections import defaultdict

# ========= Helper Functions =========

def create_ion_dict(mrm_database):
    """
    Constructs a dictionary mapping ion transitions to lipid information from the MRM database.

    Args:
        mrm_database (pd.DataFrame): DataFrame containing columns 'Parent_Ion', 'Product_Ion', 'Lipid', and 'Class'.

    Returns:
        defaultdict: A dictionary with keys as (Parent_Ion, Product_Ion) and values as a list of (Lipid, Class) tuples.
    """
    print(f"[create_ion_dict] Creating ion dictionary from MRM database with {len(mrm_database)} entries...")
    ion_dict = defaultdict(list)
    for index, row in mrm_database.iterrows():
        ion_dict[(row['Parent_Ion'], row['Product_Ion'])].append((row['Lipid'], row['Class']))
    print(f"[create_ion_dict] ✓ Created ion dictionary with {len(ion_dict)} unique ion transitions")
    return ion_dict

def within_tolerance(a, b, tolerance=0.3):
    """
    Checks if two ion values are within the specified tolerance.

    Args:
        a (float): First ion value.
        b (float): Second ion value.
        tolerance (float, optional): Tolerance threshold. Defaults to 0.3.

    Returns:
        bool: True if the absolute difference between a and b is within the tolerance, False otherwise.
    """
    return abs(a - b) <= tolerance

def match_ions(row, ion_dict, tolerance=0.3):
    """
    Matches ion values in a DataFrame row to the MRM database using the ion dictionary.

    Args:
        row (pd.Series): A row from a DataFrame containing 'Parent_Ion' and 'Product_Ion'.
        ion_dict (dict): Dictionary mapping ion transitions to lipid information.
        tolerance (float, optional): Tolerance for matching ion values. Defaults to 0.3.

    Returns:
        pd.Series: The input row updated with matched 'Lipid' and 'Class' information if a match is found.
    """
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
    """
    Parses a single mzML file to extract ion transitions and intensities.

    Processes each spectrum in the mzML file, extracting Q1 and Q3 values and summing intensities.
    Also collects OzESI chromatogram data.

    Args:
        file_path (str): Path to the mzML file.
        plot_chromatogram (bool, optional): Whether to plot the chromatogram (currently not implemented).
                                              Defaults to False.

    Side Effects:
        Updates the global DataFrames master_df and OzESI_time_df with parsed data.
    """
    global master_df, OzESI_time_df
    print(f"\n[mzml_parser] Starting to parse: {os.path.basename(file_path)}")
    rows = []
    ozesi_rows = []
    
    print(f"[mzml_parser] Opening mzML file...")
    run = pymzml.run.Reader(file_path, skip_chromatogram=False)
    q1_mz = 0
    q3_mz = 0

    spectrum_count = 0
    for spectrum in run:
        spectrum_count += 1
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
    
    print(f"[mzml_parser] Processed {spectrum_count} spectra")
    print(f"[mzml_parser] Extracted {len(rows)} transitions and {len(ozesi_rows)} OzESI data points")
    
    df = pd.DataFrame(rows)
    OzESI_time_df = pd.concat([OzESI_time_df, pd.DataFrame(ozesi_rows)], ignore_index=True)
    master_df = pd.concat([master_df, df], ignore_index=True)
    print(f'[mzml_parser] ✓ Finished parsing: {os.path.basename(file_path)}')
    print(f'[mzml_parser] Total master_df rows: {len(master_df)}, Total OzESI_time_df rows: {len(OzESI_time_df)}\n')

def mzml_parser_batch(folder_name, plot_chromatogram=False):
    """
    Parses all mzML files in the specified folder.

    Iterates over all files in the folder, and for each mzML file, calls mzml_parser.

    Args:
        folder_name (str): Path to the folder containing mzML files.
        plot_chromatogram (bool, optional): Whether to plot the chromatogram for each file.
                                            Defaults to False.
    """
    global master_df
    print(f"\n{'='*70}")
    print(f"[mzml_parser_batch] Starting batch processing of folder: {folder_name}")
    print(f"{'='*70}")
    
    data_folder = os.listdir(folder_name)
    data_folder.sort()
    mzml_files = [f for f in data_folder if f.endswith('.mzML')]
    
    print(f"[mzml_parser_batch] Found {len(mzml_files)} mzML files to process")
    
    for idx, file in enumerate(mzml_files, 1):
        print(f"\n[mzml_parser_batch] Processing file {idx}/{len(mzml_files)}")
        file_path = os.path.join(folder_name, file)
        mzml_parser(file_path, plot_chromatogram=plot_chromatogram)
    
    print(f"\n{'='*70}")
    print(f'[mzml_parser_batch] ✓ Finished parsing all {len(mzml_files)} mzML files')
    print(f"{'='*70}\n")

def save_dataframe(df, Project_results, file_name_to_save, max_attempts=5):
    """
    Saves a DataFrame to a CSV file in the specified results directory.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        Project_results (str): The results directory path (absolute or relative).
        file_name_to_save (str): The base file name for the CSV.
        max_attempts (int, optional): Maximum number of attempts to save the file if a file with the same name exists.
                                      Defaults to 5.
    """
    # Check if Project_results is an absolute path
    if os.path.isabs(Project_results):
        folder_path = Project_results
    else:
        # If relative, use the old behavior
        folder_path = f'data_results/data/data_matching/{Project_results}'
    
    print(f"[save_dataframe] Creating directory: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)
    
    for i in range(max_attempts):
        file_path = os.path.join(folder_path, f'{file_name_to_save}.csv')
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
            print(f"[save_dataframe] ✓ Saved DataFrame to: {file_path}")
            print(f"[save_dataframe] DataFrame contains {len(df)} rows and {len(df.columns)} columns")
            break
    else:
        print(f"[save_dataframe] ✗ Failed to save DataFrame after {max_attempts} attempts.")

def read_mrm_list(filename, remove_std=True, deuterated=False):
    """
    Reads and processes the MRM list from an Excel file.

    Reads all sheets from the Excel file, concatenates them, selects relevant columns, renames columns,
    rounds ion values, and optionally adjusts for deuterated compounds or removes standards.

    Args:
        filename (str): Path to the Excel file containing the MRM database.
        remove_std (bool, optional): Flag to remove standard compounds. Defaults to True.
        deuterated (bool, optional): Flag to adjust ion values for deuterated compounds. Defaults to False.

    Returns:
        pd.DataFrame: Processed MRM database with columns ['Lipid', 'Parent_Ion', 'Product_Ion', 'Class', 'Transition'].
    """
    print(f"\n[read_mrm_list] Reading MRM database from: {os.path.basename(filename)}")
    raw_mrm_data = pd.read_excel(filename, sheet_name=None)
    concatenated_mrm_data = pd.concat(raw_mrm_data, ignore_index=True)
    lipid_MRM_data = concatenated_mrm_data[['Compound Name', 'Parent Ion', 'Product Ion', 'Class']]
    lipid_MRM_data.columns = lipid_MRM_data.columns.str.replace(' ', '_')
    lipid_MRM_data['Parent_Ion'] = np.round(lipid_MRM_data['Parent_Ion'], 1)
    lipid_MRM_data['Product_Ion'] = np.round(lipid_MRM_data['Product_Ion'], 1)
    lipid_MRM_data['Transition'] = lipid_MRM_data['Parent_Ion'].astype(str) + ' -> ' + lipid_MRM_data['Product_Ion'].astype(str)
    lipid_MRM_data = lipid_MRM_data.rename(columns={'Compound_Name': 'Lipid'})
    if remove_std:
        before_count = len(lipid_MRM_data)
        lipid_MRM_data = lipid_MRM_data[~lipid_MRM_data['Lipid'].str.contains('std', case=False, na=False)]
        removed_count = before_count - len(lipid_MRM_data)
        print(f"[read_mrm_list] Removed {removed_count} standard compounds")
    
    if deuterated:
        print(f"[read_mrm_list] Adjusting for deuterated compounds (+3 to Parent and Product ions)")
        lipid_MRM_data['Parent_Ion'] = lipid_MRM_data['Parent_Ion'] + 3
        lipid_MRM_data['Product_Ion'] = lipid_MRM_data['Product_Ion'] + 3
        lipid_MRM_data['Transition'] = lipid_MRM_data['Parent_Ion'].astype(str) + ' -> ' + lipid_MRM_data['Product_Ion'].astype(str)
    
    print(f"[read_mrm_list] ✓ Final MRM database contains {len(lipid_MRM_data)} entries\n")
    return lipid_MRM_data

def match_lipids_parser(mrm_database, df, tolerance=0.3):
    """
    Matches lipids in the DataFrame against the MRM database.

    Constructs an ion dictionary from the MRM database and applies the matching function to each row in the DataFrame.

    Args:
        mrm_database (pd.DataFrame): The MRM database.
        df (pd.DataFrame): DataFrame containing parsed mzML data.
        tolerance (float, optional): Tolerance for matching ion values. Defaults to 0.3.

    Returns:
        pd.DataFrame: DataFrame with updated 'Lipid' and 'Class' columns based on matches.
    """
    print(f"\n[match_lipids_parser] Starting lipid matching with tolerance={tolerance}")
    print(f"[match_lipids_parser] Input DataFrame has {len(df)} rows")
    
    ion_dict = create_ion_dict(mrm_database)
    
    print(f"[match_lipids_parser] Matching ions against MRM database...")
    df_matched = df.apply(lambda row: match_ions(row, ion_dict, tolerance), axis=1)
    
    # Count matches
    matched_count = df_matched['Lipid'].notna().sum() if 'Lipid' in df_matched.columns else 0
    print(f"[match_lipids_parser] ✓ Matched {matched_count}/{len(df)} rows to lipids\n")
    
    return df_matched

# ========= Global DataFrame Initialization =========

# These globals are used by the mzML parsing functions.
master_df = pd.DataFrame(columns=['Parent_Ion', 'Product_Ion', 'Intensity', 'Transition', 'Sample_ID'])
OzESI_time_df = pd.DataFrame(columns=['Lipid', 'Parent_Ion', 'Product_Ion', 'Retention_Time', 
                                       'OzESI_Intensity', 'Sample_ID', 'Transition'])

# ========= The Parse Class =========

class Parse:
    """
    Encapsulates the workflow for parsing mzML files and matching lipid data.

    Attributes:
        data_base_name_location (str): Path to the MRM database Excel file.
        Project_Folder_data (str): Path to the folder containing mzML files or a single mzML file.
        Project_results (str): Folder name for saving results.
        file_name_to_save (str): Base file name for saving the CSV.
        tolerance (float): Tolerance for ion matching.
        remove_std (bool): Flag to remove standard compounds from the MRM list.
        save_data (bool): Whether to save the matched results to CSV.
        batch_processing (bool): Whether to process a folder of mzML files.
        plot_chromatogram (bool): Flag to plot chromatograms (if implemented).
    """
    def __init__(self, data_base_name_location, Project_Folder_data, Project_results, 
                 file_name_to_save, tolerance, remove_std=True, save_data=False, 
                 batch_processing=True, plot_chromatogram=False):
        """
        Initializes a new Parse instance with the given configuration.

        Args:
            data_base_name_location (str): Path to the MRM database Excel file.
            Project_Folder_data (str): Path to the folder containing mzML files or a single mzML file.
            Project_results (str): Name of the project results folder.
            file_name_to_save (str): Base file name for the CSV output.
            tolerance (float): Tolerance for matching ion values.
            remove_std (bool, optional): Flag to remove standard compounds. Defaults to True.
            save_data (bool, optional): Whether to save the results to CSV. Defaults to False.
            batch_processing (bool, optional): Whether to process a folder of mzML files. Defaults to True.
            plot_chromatogram (bool, optional): Whether to plot chromatograms. Defaults to False.
        """
        print(f"\n{'='*70}")
        print(f"[Parse.__init__] Initializing Parse object")
        print(f"{'='*70}")
        print(f"  MRM Database: {os.path.basename(data_base_name_location)}")
        print(f"  Data Location: {Project_Folder_data}")
        print(f"  Results Folder: {Project_results}")
        print(f"  Output File: {file_name_to_save}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Remove Standards: {remove_std}")
        print(f"  Save Data: {save_data}")
        print(f"  Batch Processing: {batch_processing}")
        print(f"{'='*70}\n")
        
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
        Reads the MRM database using the instance's configuration.

        Args:
            deuterated (bool, optional): Flag to adjust for deuterated compounds. Defaults to False.

        Returns:
            pd.DataFrame: The processed MRM database.
        """
        mrm_database = read_mrm_list(self.data_base_name_location, remove_std=self.remove_std, deuterated=deuterated)
        return mrm_database

    def match_lipids_parser(self, mrm_database, df):
        """
        Matches lipids in the provided DataFrame against the MRM database using the specified tolerance.

        Args:
            mrm_database (pd.DataFrame): The MRM database.
            df (pd.DataFrame): DataFrame containing parsed mzML or OzESI data.

        Returns:
            pd.DataFrame: DataFrame with matched lipid information.
        """
        return match_lipids_parser(mrm_database, df, tolerance=self.tolerance)

    def full_parse(self):
        """
        Executes the complete parsing workflow:
            1. Resets the global DataFrames.
            2. Reads the MRM database.
            3. Parses mzML files (either in batch mode or as a single file).
            4. Matches parsed data against the MRM database.
            5. Optionally saves the matched DataFrame.

        Returns:
            tuple: (df_matched, OzESI_time_df)
                - df_matched: Matched master DataFrame.
                - OzESI_time_df: DataFrame containing OzESI chromatogram data.
        """
        global master_df, OzESI_time_df
        
        print(f"\n{'#'*70}")
        print(f"# STARTING FULL_PARSE WORKFLOW")
        print(f"{'#'*70}\n")
        
        # (Optional) Reset the global DataFrames before processing
        print(f"[full_parse] Step 1: Resetting global DataFrames...")
        master_df = pd.DataFrame(columns=['Parent_Ion', 'Product_Ion', 'Intensity', 'Transition', 'Sample_ID'])
        OzESI_time_df = pd.DataFrame(columns=['Lipid', 'Parent_Ion', 'Product_Ion', 'Retention_Time', 
                                               'OzESI_Intensity', 'Sample_ID', 'Transition'])
        print(f"[full_parse] ✓ DataFrames reset\n")
        
        # Step 1: Read the MRM database.
        print(f"[full_parse] Step 2: Reading MRM database...")
        mrm_database = self.read_mrm_list()
        
        # Step 2: Parse mzML files.
        print(f"[full_parse] Step 3: Parsing mzML files...")
        if self.batch_processing:
            mzml_parser_batch(self.Project_Folder_data, plot_chromatogram=self.plot_chromatogram)
        else:
            mzml_parser(self.Project_Folder_data, plot_chromatogram=self.plot_chromatogram)
        
        # Step 3: Match lipids using the parsed master_df.
        print(f"[full_parse] Step 4: Matching lipids from parsed data...")
        df_matched = self.match_lipids_parser(mrm_database, master_df)
        
        # Step 4: Optionally save the matched DataFrame.
        if self.save_data:
            print(f"[full_parse] Step 5: Saving matched DataFrame...")
            save_dataframe(df_matched, self.Project_results, self.file_name_to_save)
        else:
            print(f"[full_parse] Step 5: Skipping save (save_data=False)")
        
        print(f"\n[full_parse] ✓ Full parse workflow complete!")
        print(f"[full_parse] Returning {len(df_matched)} matched rows and {len(OzESI_time_df)} OzESI rows\n")
        
        return df_matched, OzESI_time_df

    def mrm_run_all(self, deuterated=False):
        """
        Runs the complete workflow including a second matching step for OzESI chromatogram data:
            1. Executes full_parse to process mzML files and perform the initial matching.
            2. Re-reads the MRM database (with an option for deuterated adjustments).
            3. Matches the OzESI chromatogram data against the MRM database.

        Args:
            deuterated (bool, optional): Flag to adjust for deuterated compounds when re-reading the MRM database.
                                         Defaults to False.

        Returns:
            tuple: (df_full_matched, OzESI_time_df, df_OzESI_matched)
                - df_full_matched: Matched master DataFrame from full_parse.
                - OzESI_time_df: DataFrame containing OzESI chromatogram data.
                - df_OzESI_matched: OzESI DataFrame with updated matched lipid information.
        """
        print(f"\n{'#'*70}")
        print(f"# STARTING MRM_RUN_ALL WORKFLOW")
        print(f"{'#'*70}\n")
        
        # Step 1: Run the full parse workflow.
        print(f"[mrm_run_all] Step 1: Running full_parse workflow...\n")
        df_full_matched, ozesi_time_df = self.full_parse()
        
        # Step 2: Re-read the MRM database (with the provided deuterated flag).
        print(f"\n[mrm_run_all] Step 2: Re-reading MRM database for OzESI matching (deuterated={deuterated})...")
        mrm_database = self.read_mrm_list(deuterated=deuterated)
        
        # Step 3: Match the OzESI chromatogram data using the re-read MRM database.
        print(f"[mrm_run_all] Step 3: Matching OzESI chromatogram data...")
        df_OzESI_matched = self.match_lipids_parser(mrm_database, ozesi_time_df)
        
        # Save each df to csv file before returning
        print(f"\n[mrm_run_all] Step 4: Saving all DataFrames to CSV...")
        print(f"[mrm_run_all] Saving df_full_matched.csv...")
        df_full_matched.to_csv('df_full_matched.csv', index=False)
        print(f"[mrm_run_all] ✓ Saved df_full_matched.csv ({len(df_full_matched)} rows)")
        
        print(f"[mrm_run_all] Saving ozesi_time_df.csv...")
        ozesi_time_df.to_csv('ozesi_time_df.csv', index=False)
        print(f"[mrm_run_all] ✓ Saved ozesi_time_df.csv ({len(ozesi_time_df)} rows)")
        
        print(f"[mrm_run_all] Saving df_OzESI_matched.csv...")
        df_OzESI_matched.to_csv('df_OzESI_matched.csv', index=False)
        print(f"[mrm_run_all] ✓ Saved df_OzESI_matched.csv ({len(df_OzESI_matched)} rows)")
        
        print(f"\n{'#'*70}")
        print(f"# MRM_RUN_ALL WORKFLOW COMPLETE!")
        print(f"{'#'*70}\n")
        
        return df_full_matched, ozesi_time_df, df_OzESI_matched
