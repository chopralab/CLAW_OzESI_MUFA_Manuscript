#!/usr/bin/env python
"""
Script: CLAW_OzESI_filter.py

Description:
    This module processes OzESI (Ozone Electrospray Ionization) data derived from mass spectrometry
    experiments by applying a series of filtering and matching steps. The workflow includes:
      1. Filtering based on retention time and intensity thresholds.
      2. Handling special cases for specific sample IDs and parent ion values.
      3. Calculating double bond (DB) positions based on parent ion values.
      4. Matching lipid information by comparing calculated DB positions with measured values.
      5. Calculating intensity ratios between different lipid fragments (e.g., n-9 and n-7).
      6. Sorting and selecting the highest intensity values within groups.

    **Note:** All internal operations use CSV strings. The `run_all` function now accepts a filename,
    loads the CSV from that file, and returns a message indicating where the final CSV was saved.

Usage:
    Import this module and use the `OzESIFilter` class. For example:

        from CLAW_OzESI_filter import OzESIFilter

        # Provide the CSV filename as input.
        ozesi_filter = OzESIFilter(tolerance=0.3)
        message = ozesi_filter.run_all(
            df_csv="df_OzESI_matched.csv",
            min_rt=10,
            max_rt=23,
            min_intensity=100,
            db_pos_list="7,9,12",
            sort_by_columns="Sample_ID,Product_Ion",
            tolerance=0.3,
            save_csv_path="df5_csv.csv"  # Optional: provide a path to save the final CSV.
        )
        print(message)

Author: Your Name or Organization
Date: YYYY-MM-DD
"""

import pandas as pd
import numpy as np
from io import StringIO

class OzESIFilter:
    def __init__(self, tolerance: float = 0.3) -> None:
        """
        Initializes an OzESIFilter instance.
        
        Parameters:
            tolerance (float): Default tolerance value for matching ions.
        """
        self.tolerance = tolerance

    def within_tolerance(self, a: float, b: float, tolerance: float = None) -> bool:
        """
        Checks if the absolute difference between two values is within the given tolerance.
        
        Parameters:
            a (float): The first value.
            b (float): The second value.
            tolerance (float, optional): Tolerance for the comparison. Defaults to instance's tolerance if None.
            
        Returns:
            bool: True if |a - b| <= tolerance, else False.
        """
        if tolerance is None:
            tolerance = self.tolerance
        return abs(a - b) <= tolerance

    def filter_rt(self, df_csv: str, min_rt: float = 10.0, max_rt: float = 22.0, min_intensity: int = 100,
                  special_min_rt: float = 19.5, special_max_rt: float = 21.5,
                  special_sample_id: str = 'DegummedCanola_O3on_150gN3_02082023', special_parent_ion: float = 794.6) -> str:
        """
        Filters the CSV string (DataFrame) based on retention time and OzESI intensity.
        
        Parameters:
            df_csv (str): CSV string representing the DataFrame with OzESI data.
            min_rt (float): Minimum retention time.
            max_rt (float): Maximum retention time.
            min_intensity (int): Minimum OzESI intensity.
            special_min_rt (float): Lower bound for special case retention time.
            special_max_rt (float): Upper bound for special case retention time.
            special_sample_id (str): Specific sample ID for special filter.
            special_parent_ion (float): Specific parent ion for special filter.
        
        Returns:
            str: CSV string representing the filtered and aggregated DataFrame.
        """
        df = pd.read_csv(StringIO(df_csv))
        general_filter = (df['Retention_Time'] > min_rt) & (df['Retention_Time'] < max_rt) & (df['OzESI_Intensity'] > min_intensity)
        filtered_df = df[general_filter].copy()
        special_case_filter = (
            (df['Sample_ID'] == special_sample_id) &
            (df['Parent_Ion'] == special_parent_ion) &
            (df['Retention_Time'] >= special_min_rt) &
            (df['Retention_Time'] <= special_max_rt)
        )
        filtered_df['is_special_case'] = special_case_filter.astype(int)
        filtered_df['Retention_Time'] = filtered_df['Retention_Time'].round(2)
        filtered_df['OzESI_Intensity'] = filtered_df['OzESI_Intensity'].round(0)

        def apply_aggregation(group: pd.DataFrame) -> pd.Series:
            if group['is_special_case'].sum() > 0:
                special_case_rows = group[group['is_special_case'] == 1]
                return special_case_rows.loc[special_case_rows['OzESI_Intensity'].idxmax()]
            else:
                return group.loc[group['OzESI_Intensity'].idxmax()]

        result_df = filtered_df.groupby(['Sample_ID', 'Transition']).apply(apply_aggregation).reset_index(drop=True)
        return result_df.to_csv(index=False)

    def calculate_DB_Position(self, df_matched_ions_csv: str, db_pos_list: str = "7,9,12") -> str:
        """
        Calculates double bond positions and adds new columns to the CSV string (DataFrame).
        
        Parameters:
            df_matched_ions_csv (str): CSV string representing the DataFrame with matched ion data.
            db_pos_list (str): Comma-separated string of DB positions (e.g., "7,9,12").
            
        Returns:
            str: CSV string representing the updated DataFrame with additional n-# columns.
        """
        df = pd.read_csv(StringIO(df_matched_ions_csv))
        # Create lookup table for Aldehyde_Ion values
        df_DB_aldehyde = pd.DataFrame(columns=['DB_Position', 'Aldehyde_Ion'])
        for position in range(3, 21):
            df_DB_aldehyde.loc[position, 'DB_Position'] = position
            df_DB_aldehyde.loc[position, 'Aldehyde_Ion'] = 26 + (14 * (position - 3))
        # Convert db_pos_list string to list of integers
        positions = [int(x.strip()) for x in db_pos_list.split(',')]
        for pos in positions:
            aldehyde_ion = df_DB_aldehyde.loc[df_DB_aldehyde["DB_Position"] == pos, "Aldehyde_Ion"].values[0]
            df["n-" + str(pos)] = df["Parent_Ion"] - aldehyde_ion
        return df.to_csv(index=False)

    def add_lipid_info(self, matched_dataframe_csv: str, db_pos: str, tolerance: float = None) -> str:
        """
        Matches and adds lipid information based on DB positions.
        
        For rows missing lipid information, attempts are made to match the Parent_Ion to a calculated n-#
        value (using the provided tolerance). When a match is found, the 'Lipid' field is updated.
        
        Parameters:
            matched_dataframe_csv (str): CSV string representing the DataFrame with matched ion data and calculated n-# columns.
            db_pos (str): Comma-separated string of DB positions (e.g., "7,9,12").
            tolerance (float, optional): Tolerance for matching; uses the instance's tolerance if None.
        
        Returns:
            str: CSV string representing the updated DataFrame with updated lipid information and new rows added.
        """
        df = pd.read_csv(StringIO(matched_dataframe_csv))
        # Convert db_pos string to list of integers
        db_pos_list = [int(x.strip()) for x in db_pos.split(',')]
        
        working_dataframe = df.copy()
        final_dataframe = df.copy()
        
        # Ensure the n-# columns are float type
        for pos in db_pos_list:
            col = 'n-' + str(pos)
            if col in working_dataframe.columns:
                working_dataframe[col] = working_dataframe[col].astype(float)
                
        new_rows = []
        for i in range(len(working_dataframe)):
            if pd.isna(working_dataframe.loc[i, 'Lipid']):
                parent_ion = working_dataframe.loc[i, 'Parent_Ion']
                for j in range(len(working_dataframe)):
                    current_row = working_dataframe.loc[j].copy()
                    for n in [9, 7]:
                        col_name = f'n-{n}'
                        if col_name in working_dataframe.columns and isinstance(current_row.get('Lipid', None), str):
                            if self.within_tolerance(parent_ion, current_row[col_name], tolerance):
                                working_dataframe.loc[i, 'Lipid'] = current_row['Lipid']
                                current_db_pos = current_row.get('db_pos', '')
                                working_dataframe.loc[i, 'db_pos'] = f'n-{n}' + (current_db_pos if pd.notna(current_db_pos) else '')
                                appended_row = working_dataframe.loc[i].copy()
                                appended_row['db_pos'] = f'n-{n}' + (current_db_pos if pd.notna(current_db_pos) else '')
                                new_rows.append(appended_row)
        if new_rows:
            final_dataframe = pd.concat([final_dataframe, pd.DataFrame(new_rows)], ignore_index=True)
        final_dataframe.dropna(subset=['Lipid'], inplace=True)
        return final_dataframe.to_csv(index=False)

    def calculate_intensity_ratio(self, df_csv: str) -> str:
        """
        Calculates intensity ratios by comparing n-9 and n-7 intensities.
        
        Parameters:
            df_csv (str): CSV string representing the DataFrame with OzESI data and lipid information.
            
        Returns:
            str: CSV string representing the updated DataFrame with a new 'Ratio' column.
        """
        df = pd.read_csv(StringIO(df_csv))
        df['Ratio'] = pd.Series(dtype='float64')
        for index, row in df.iterrows():
            lipid = row['Lipid']
            label = row['db_pos']
            intensity = row['OzESI_Intensity']
            sample_id = row['Sample_ID']
            if label == 'n-9':
                if lipid == 'TG(54:2)_FA18:1':
                    n7_row = df[(df['Lipid'] == lipid) &
                                (df['db_pos'] == 'n-7') &
                                (df['Sample_ID'] == sample_id) &
                                (df['Retention_Time'].between(19.2, 21.5))]
                else:
                    n7_row = df[(df['Lipid'] == lipid) &
                                (df['db_pos'] == 'n-7') &
                                (df['Sample_ID'] == sample_id)]
                if not n7_row.empty:
                    n7_intensity = n7_row['OzESI_Intensity'].values[0]
                    ratio = intensity / n7_intensity
                    df.at[index, 'Ratio'] = ratio
        return df.to_csv(index=False)

    def sort_by_second_tg(self, lipid: str) -> str:
        """
        Extracts the second triglyceride component from a lipid name if present.
        
        Parameters:
            lipid (str): Lipid name.
            
        Returns:
            str: Second component if available, else original lipid name.
        """
        if pd.isna(lipid):
            return lipid
        tgs = lipid.split(',')
        if len(tgs) > 1:
            return tgs[1].strip()
        else:
            return lipid

    def filter_highest_ratio(self, df_csv: str) -> str:
        """
        Filters the CSV string (DataFrame) to retain only the rows with the highest intensity ratio for each group.
        
        Parameters:
            df_csv (str): CSV string representing the DataFrame with a 'Ratio' column.
            
        Returns:
            str: CSV string representing the filtered DataFrame.
        """
        df = pd.read_csv(StringIO(df_csv))
        df_sorted = df.sort_values(by='Ratio', ascending=False)
        df_filtered = df_sorted.drop_duplicates(subset=['Sample_ID', 'Lipid', 'db_pos'], keep='first')
        df_filtered = df_filtered.sort_values(by=['Sample_ID', 'Lipid'], ascending=[True, True])
        return df_filtered.to_csv(index=False)

    def run_all(self, df_csv: str, min_rt: float = 10, max_rt: float = 23, min_intensity: float = 100,
                db_pos_list: str = "7,9,12", sort_by_columns: str = "Sample_ID,Product_Ion",
                save_csv_path: str = None, tolerance: float = None) -> str:
        """
        Executes the complete filtering workflow.
        
        Parameters:
            df_csv (str): **Filename** of the input OzESI matched CSV file.
            min_rt (float): Minimum retention time.
            max_rt (float): Maximum retention time.
            min_intensity (float): Minimum OzESI intensity.
            db_pos_list (str): Comma-separated string of DB positions (e.g., "7,9,12").
            sort_by_columns (str): Comma-separated columns to sort by (e.g., "Sample_ID,Product_Ion").
            save_csv_path (str): File path to save the final CSV (optional).
            tolerance (float): Tolerance for matching lipid info.
            
        Returns:
            str: A message indicating where the final CSV file was saved.
        """
        # Load the CSV file content from the provided filename.
        with open(df_csv, 'r') as file:
            csv_content = file.read()
        
        # Step 1: Filter based on retention time and intensity.
        df1_csv = self.filter_rt(csv_content, min_rt=min_rt, max_rt=max_rt, min_intensity=min_intensity)
        
        # Step 2: Calculate double bond positions.
        df2_csv = self.calculate_DB_Position(df1_csv, db_pos_list=db_pos_list)
        # Add a new column for DB label.
        df2 = pd.read_csv(StringIO(df2_csv))
        df2['db_pos'] = ''
        df2_csv = df2.to_csv(index=False)
        
        # Step 3: Match DB positions to add lipid information.
        df3_csv = self.add_lipid_info(df2_csv, db_pos=db_pos_list, tolerance=tolerance)
        
        # Step 4: Sort by provided columns.
        sort_columns = [col.strip() for col in sort_by_columns.split(',')]
        df3 = pd.read_csv(StringIO(df3_csv))
        df3_sorted = df3.sort_values(by=sort_columns)
        df3_csv = df3_sorted.to_csv(index=False)
        
        # Step 5: Calculate intensity ratios.
        df4_csv = self.calculate_intensity_ratio(df3_csv)
        
        # Step 6: Adjust lipid names by extracting second TG component.
        df4 = pd.read_csv(StringIO(df4_csv))
        df4['Lipid'] = df4['Lipid'].apply(self.sort_by_second_tg)
        df4_csv = df4.to_csv(index=False)
        
        # Step 7: Filter to retain only the rows with highest intensity ratios.
        df5_csv = self.filter_highest_ratio(df4_csv)
        
        # Save the final CSV to the specified file if provided; otherwise, save to 'df5csv.csv'
        if save_csv_path is not None and save_csv_path.strip() != "":
            final_filename = save_csv_path
            final_df = pd.read_csv(StringIO(df5_csv))
            final_df.to_csv(final_filename, index=False)
        else:
            final_filename = "df5csv.csv"
        
        # Also always save to 'df5csv.csv' in the current directory.
        with open("df5csv.csv", "w") as f:
            f.write(df5_csv)
        
        return f"CSV file successfully saved to {final_filename}"
