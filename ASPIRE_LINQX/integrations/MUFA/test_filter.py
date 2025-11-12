#!/usr/bin/env python
"""Test script to run CLAW_OzESI_filter.py with parsed_results.csv"""

import sys
sys.path.insert(0, '/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/scripts')

from CLAW_OzESI_filter import OzESIFilter

# Initialize the filter
ozesi_filter = OzESIFilter(tolerance=0.3)

# Run the filter with the parsed_results.csv
try:
    message = ozesi_filter.run_all(
        df_csv="/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results/df_OzESI_matched.csv",
        min_rt=10,
        max_rt=23,
        min_intensity=100,
        db_pos_list="7,9,12",
        sort_by_columns="Sample_ID,Product_Ion",
        tolerance=0.3,
        save_csv_path="/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/agent_results/filtered_results.csv"
    )
    print(message)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
