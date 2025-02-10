import os
import pandas as pd
import numpy as np
import pymzml
from collections import defaultdict

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