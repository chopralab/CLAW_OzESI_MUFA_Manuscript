import pandas as pd
import numpy as np

class OzESIFilter:
    def __init__(self, tolerance=0.3):
        """
        Initializes an OzESIFilter instance.

        Parameters:
            tolerance (float): Default tolerance value for matching ions.
        """
        self.tolerance = tolerance

    def within_tolerance(self, a, b, tolerance=None):
        """
        Checks if the absolute difference between two values is within the given tolerance.
        """
        if tolerance is None:
            tolerance = self.tolerance
        return abs(a - b) <= tolerance

    def filter_rt(self, df, min_rt=10.0, max_rt=22.0, min_intensity=100, 
                  special_min_rt=19.5, special_max_rt=21.5, 
                  special_sample_id='DegummedCanola_O3on_150gN3_02082023', 
                  special_parent_ion=794.6):
        """
        Filters the DataFrame based on retention times and OzESI intensity.
        Aggregates by maximum intensity for each unique combination of 'Sample_ID' and 'Transition'.
        Applies a special case if specific criteria are met.
        """
        # General filter based on retention time and intensity
        general_filter = (df['Retention_Time'] > min_rt) & \
                         (df['Retention_Time'] < max_rt) & \
                         (df['OzESI_Intensity'] > min_intensity)
        filtered_df = df[general_filter].copy()

        # Special case filter
        special_case_filter = (
            (df['Sample_ID'] == special_sample_id) & 
            (df['Parent_Ion'] == special_parent_ion) & 
            (df['Retention_Time'] >= special_min_rt) & 
            (df['Retention_Time'] <= special_max_rt)
        )
        filtered_df['is_special_case'] = special_case_filter.astype(int)
        filtered_df['Retention_Time'] = filtered_df['Retention_Time'].round(2)
        filtered_df['OzESI_Intensity'] = filtered_df['OzESI_Intensity'].round(0)

        # Within each group, take the row with the maximum OzESI_Intensity.
        def apply_aggregation(group):
            if group['is_special_case'].sum() > 0:
                special_case_rows = group[group['is_special_case'] == 1]
                return special_case_rows.loc[special_case_rows['OzESI_Intensity'].idxmax()]
            else:
                return group.loc[group['OzESI_Intensity'].idxmax()]

        result_df = filtered_df.groupby(['Sample_ID', 'Transition']).apply(apply_aggregation).reset_index(drop=True)
        return result_df

    def calculate_DB_Position(self, df_matched_ions, db_pos_list=[7, 9, 12]):
        """
        Calculates double bond (DB) positions by subtracting a calculated Aldehyde_Ion value from the Parent_Ion.
        Adds new columns for each n-# value.
        """
        df_DB_aldehyde = pd.DataFrame(columns=['DB_Position', 'Aldehyde_Ion'])
        for position in range(3, 21):
            df_DB_aldehyde.loc[position, 'DB_Position'] = position
            df_DB_aldehyde.loc[position, 'Aldehyde_Ion'] = 26 + (14 * (position - 3))
            
        for ozesi_position in db_pos_list:
            aldehyde_ion = df_DB_aldehyde.loc[df_DB_aldehyde["DB_Position"] == ozesi_position, "Aldehyde_Ion"].values[0]
            df_matched_ions["n-{}".format(ozesi_position)] = df_matched_ions["Parent_Ion"] - aldehyde_ion
        return df_matched_ions

    def add_lipid_info(self, matched_dataframe, db_pos, tolerance=None):
        """
        Matches and adds lipid information based on DB positions.
        If the 'Lipid' field is missing, attempts to fill it in by checking if the Parent_Ion is within tolerance
        of a calculated n-# value.
        """
        if tolerance is None:
            tolerance = self.tolerance

        working_dataframe = matched_dataframe.copy()
        final_dataframe = matched_dataframe.copy()
        
        for position in db_pos:
            working_dataframe['n-' + str(position)] = working_dataframe['n-' + str(position)].astype(float)
        
        new_rows = []  # Store additional rows with matched lipid information
        for i in range(len(working_dataframe)):
            if pd.isna(working_dataframe.loc[i, 'Lipid']):
                parent_ion = working_dataframe.loc[i, 'Parent_Ion']
                for j in range(len(working_dataframe)):
                    current_row = working_dataframe.loc[j].copy()
                    # Check for a match for n-9
                    for n in [9]:
                        if self.within_tolerance(parent_ion, current_row[f'n-{n}'], tolerance) and isinstance(current_row['Lipid'], str):
                            working_dataframe.loc[i, 'Lipid'] = current_row['Lipid']
                            working_dataframe.loc[i, 'db_pos'] = f'n-{n}' + current_row['db_pos']
                            appended_row = working_dataframe.loc[i].copy()
                            appended_row['db_pos'] = f'n-{n}' + current_row['db_pos']
                            new_rows.append(appended_row)
                    # Check for a match for n-7
                    for n in [7]:
                        if self.within_tolerance(parent_ion, current_row[f'n-{n}'], tolerance) and isinstance(current_row['Lipid'], str):
                            working_dataframe.loc[i, 'Lipid'] = current_row['Lipid']
                            working_dataframe.loc[i, 'db_pos'] = f'n-{n}' + current_row['db_pos']
                            appended_row = working_dataframe.loc[i].copy()
                            appended_row['db_pos'] = f'n-{n}' + current_row['db_pos']
                            new_rows.append(appended_row)
                        
        if new_rows:
            final_dataframe = pd.concat([final_dataframe, pd.DataFrame(new_rows)], ignore_index=True)
        
        final_dataframe.dropna(subset=['Lipid'], inplace=True)
        return final_dataframe

    def calculate_intensity_ratio(self, df):
        """
        Calculates the intensity ratio for each lipid based on the OzESI intensity.
        The ratio is computed as the intensity of the n-9 label divided by that of the n-7 label.
        """
        df['Ratio'] = pd.Series(dtype='float64')
        for index, row in df.iterrows():
            lipid = row['Lipid']
            label = row['db_pos']
            intensity = row['OzESI_Intensity']
            sample_id = row['Sample_ID']
            retention_time = row['Retention_Time']
            
            # Only process rows with label n-9
            if label == 'n-9':
                if lipid == 'TG(54:2)_FA18:1':
                    # Special case for TG(54:2)_FA18:1
                    n7_row = df[
                        (df['Lipid'] == lipid) &
                        (df['db_pos'] == 'n-7') &
                        (df['Sample_ID'] == sample_id) &
                        (df['Retention_Time'].between(19.2, 21.5))
                    ]
                else:
                    n7_row = df[
                        (df['Lipid'] == lipid) &
                        (df['db_pos'] == 'n-7') &
                        (df['Sample_ID'] == sample_id)
                    ]
                if not n7_row.empty:
                    n7_intensity = n7_row['OzESI_Intensity'].values[0]
                    ratio = intensity / n7_intensity
                    df.at[index, 'Ratio'] = ratio
        return df

    def sort_by_second_tg(self, lipid):
        """
        Returns the second triglyceride component from a lipid name if present.
        """
        if pd.isna(lipid):
            return lipid
        tgs = lipid.split(',')
        if len(tgs) > 1:
            return tgs[1]
        else:
            return lipid

    def filter_highest_ratio(self, df):
        """
        Filters the DataFrame to retain only the rows with the highest intensity ratio
        for each unique combination of Sample_ID, Lipid, and db_pos.
        """
        df_sorted = df.sort_values(by='Ratio', ascending=False)
        df_filtered = df_sorted.drop_duplicates(subset=['Sample_ID', 'Lipid', 'db_pos'], keep='first')
        df_filtered = df_filtered.sort_values(by=['Sample_ID', 'Lipid'], ascending=[True, True])
        return df_filtered

    def run_all(self, df, min_rt=10, max_rt=23, min_intensity=100, db_pos_list=[7, 9, 12],
                sort_by_columns=['Sample_ID', 'Product_Ion'], save_csv_path=None, tolerance=None):
        """
        Runs all filtering steps sequentially on the input DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input OzESI matched DataFrame.
            min_rt (float): Minimum retention time.
            max_rt (float): Maximum retention time.
            min_intensity (float): Minimum OzESI intensity.
            db_pos_list (list): List of double bond positions to calculate.
            sort_by_columns (list): List of columns to sort on after adding DB positions.
            save_csv_path (str): If provided, the final DataFrame will be saved to this path.
            tolerance (float): Tolerance for matching lipid info; defaults to self.tolerance if not provided.
            
        Returns:
            pd.DataFrame: Final processed DataFrame.
        """
        if tolerance is None:
            tolerance = self.tolerance

        # Step 1: Filter based on retention time and intensity.
        df1 = self.filter_rt(df, min_rt=min_rt, max_rt=max_rt, min_intensity=min_intensity)
        df1_copy = df1.copy()

        # Step 2: Calculate double bond positions.
        df2 = self.calculate_DB_Position(df1_copy, db_pos_list=db_pos_list)
        # Create a new column for n-# labels.
        df2['db_pos'] = ''

        # Step 3: Match DB positions to n-# labels.
        df3 = self.add_lipid_info(df2, db_pos_list, tolerance=tolerance)

        # Step 4: Sort by provided columns.
        df3_sorted = df3.sort_values(by=sort_by_columns)

        # Step 5: Calculate intensity ratios.
        df4 = df3_sorted.copy()
        df4 = self.calculate_intensity_ratio(df4)

        # Step 6: Sort Lipid names by second TG component.
        df4['Lipid'] = df4['Lipid'].apply(self.sort_by_second_tg)

        # Step 7: Filter to retain only rows with highest intensity ratios.
        df5 = self.filter_highest_ratio(df4)

        # Optionally save the final DataFrame to CSV.
        if save_csv_path is not None:
            df5.to_csv(save_csv_path, index=False)

        return df5
