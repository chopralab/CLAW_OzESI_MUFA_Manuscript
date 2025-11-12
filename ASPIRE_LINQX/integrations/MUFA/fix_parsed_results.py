#!/usr/bin/env python
"""
Script to fix parsed_results.csv by adding Retention_Time column and renaming Intensity column.
"""

import pandas as pd
import re

# Load the files
parsed_results = pd.read_csv('/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/agent_results/parsed_results.csv')
lipid_rt_map = pd.read_csv('/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results/lipid_RT_map.csv')

print("Original parsed_results.csv columns:", parsed_results.columns.tolist())
print("Rows in parsed_results:", len(parsed_results))
print("\nLipid RT map:")
print(lipid_rt_map)

# Rename Intensity to OzESI_Intensity
parsed_results = parsed_results.rename(columns={'Intensity': 'OzESI_Intensity'})

# Extract the TG part from Lipid column to match with lipid_rt_map
# Example: "[TG(52:6)]_FA18:1" -> "TG(52:6)"
def extract_tg(lipid_str):
    if pd.isna(lipid_str):
        return None
    # Try to extract TG pattern
    match = re.search(r'TG\(\d+:\d+\)', lipid_str)
    if match:
        return match.group(0)
    return None

parsed_results['TG_extracted'] = parsed_results['Lipid'].apply(extract_tg)

# Create a mapping dictionary from lipid_rt_map
rt_mapping = dict(zip(lipid_rt_map['Lipid'], lipid_rt_map['RT']))

# Map the retention times
parsed_results['Retention_Time'] = parsed_results['TG_extracted'].map(rt_mapping)

# Drop the temporary column
parsed_results = parsed_results.drop(columns=['TG_extracted'])

# Check how many rows got retention times
print("\nRows with Retention_Time assigned:", parsed_results['Retention_Time'].notna().sum())
print("Rows without Retention_Time:", parsed_results['Retention_Time'].isna().sum())

# Show sample of results
print("\nSample of fixed data:")
print(parsed_results[['Lipid', 'OzESI_Intensity', 'Retention_Time', 'Sample_ID']].head(10))

# Save the fixed file
output_path = '/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/agent_results/parsed_results_fixed.csv'
parsed_results.to_csv(output_path, index=False)
print(f"\n✅ Fixed CSV saved to: {output_path}")
