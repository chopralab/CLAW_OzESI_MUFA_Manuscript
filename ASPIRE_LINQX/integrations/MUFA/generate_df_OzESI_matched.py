#!/usr/bin/env python
"""
Script to generate df_OzESI_matched.csv from CLAW_OzESI.py

This script demonstrates how to use the Parse class from CLAW_OzESI.py
to process mzML files and generate the df_OzESI_matched.csv file.
"""

import sys
sys.path.insert(0, '/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/scripts')

from CLAW_OzESI import Parse

# ========= Configuration =========

# Path to the MRM database Excel file containing lipid information
data_base_name_location = "/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/lipid_database/Lipid_Database.xlsx"

# Path to folder containing mzML files to parse
Project_Folder_data = "/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/Projects/canola/mzml/ON"

# Path to save results
Project_results = "/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results"

# Base filename for saving CSV files
file_name_to_save = "parsed_results"

# Ion matching tolerance
tolerance = 0.3

# ========= Initialize Parse Object =========

print("\n" + "="*70)
print("GENERATING df_OzESI_matched.csv")
print("="*70 + "\n")

parser = Parse(
    data_base_name_location=data_base_name_location,
    Project_Folder_data=Project_Folder_data,
    Project_results=Project_results,
    file_name_to_save=file_name_to_save,
    tolerance=tolerance,
    remove_std=True,          # Remove standard compounds from MRM database
    save_data=True,           # Save the matched results to CSV
    batch_processing=True,    # Process all mzML files in the folder
    plot_chromatogram=False   # Don't plot chromatograms
)

# ========= Run the Full Workflow =========

# The mrm_run_all() method:
# 1. Parses all mzML files in the folder
# 2. Matches ions against the MRM database
# 3. Creates OzESI chromatogram data with retention times
# 4. Matches lipids in the OzESI data
# 5. Saves three CSV files:
#    - df_full_matched.csv (master matched data)
#    - ozesi_time_df.csv (raw OzESI chromatogram data)
#    - df_OzESI_matched.csv (OzESI data with matched lipids) ← This is what we need!

df_full_matched, ozesi_time_df, df_OzESI_matched = parser.mrm_run_all(deuterated=False)

# ========= Summary =========

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✅ df_full_matched:    {len(df_full_matched)} rows")
print(f"✅ ozesi_time_df:      {len(ozesi_time_df)} rows")
print(f"✅ df_OzESI_matched:   {len(df_OzESI_matched)} rows")
print("\n📁 Files saved in current directory:")
print("   - df_full_matched.csv")
print("   - ozesi_time_df.csv")
print("   - df_OzESI_matched.csv  ← This is the input for CLAW_OzESI_filter.py")
print("="*70 + "\n")

print("✨ df_OzESI_matched.csv has been generated successfully!")
print("\nThis file contains:")
print("  - Lipid: Matched lipid names")
print("  - Parent_Ion: Parent ion m/z")
print("  - Product_Ion: Product ion m/z")
print("  - Retention_Time: Chromatographic retention time")
print("  - OzESI_Intensity: Intensity at each time point")
print("  - Sample_ID: Sample identifier")
print("  - Transition: Ion transition string")
print("  - Class: Lipid class (TAG, DAG, etc.)")
