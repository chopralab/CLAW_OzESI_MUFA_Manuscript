# Complete OzESI Data Processing Workflow

## Overview

This document explains the complete workflow from raw mzML files to filtered, processed lipid data.

---

## 📊 Workflow Diagram

```
mzML files
    ↓
[CLAW_OzESI.py]
    ↓
df_OzESI_matched.csv (225,359 rows with Retention_Time & OzESI_Intensity)
    ↓
[CLAW_OzESI_filter.py]
    ↓
filtered_results.csv (98 rows with ratios and DB positions)
```

---

## Step 1: Generate df_OzESI_matched.csv from mzML files

### Input Files
- **mzML files**: Raw mass spectrometry data files
  - Location: `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/Projects/canola/mzml/ON`
  - Files:
    - `CrudeCanola_O3on_150gN3_02082023.mzML`
    - `DegummedCanola_O3on_150gN3_02082023.mzML`
    - `RBDCanola_O3on_150gN3_02082023.mzML`

- **MRM Database**: Excel file with lipid reference data
  - Location: `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/lipid_database/Lipid_Database.xlsx`
  - Contains: Parent ions, Product ions, Lipid names, Classes

### Script: `CLAW_OzESI.py`

**Key Class**: `Parse`

**Key Method**: `mrm_run_all()`

This method performs:
1. **Parse mzML files** - Extract ion transitions, intensities, retention times
2. **Match against MRM database** - Identify lipids based on ion transitions
3. **Generate chromatogram data** - Create time-series data with all data points
4. **Save results** - Export three CSV files

### Output Files

The `mrm_run_all()` method creates **3 CSV files**:

1. **`df_full_matched.csv`** (105 rows)
   - Summarized intensity per transition
   - Columns: `Class, Intensity, Lipid, Parent_Ion, Product_Ion, Sample_ID, Transition`

2. **`ozesi_time_df.csv`** (225,357 rows)
   - Raw time-series chromatogram data (before lipid matching)
   - Columns: `Parent_Ion, Product_Ion, Retention_Time, OzESI_Intensity, Sample_ID, Transition`

3. **`df_OzESI_matched.csv`** (225,359 rows) ⭐ **This is the key file**
   - Time-series chromatogram data WITH matched lipid information
   - Columns: `Class, Lipid, OzESI_Intensity, Parent_Ion, Product_Ion, Retention_Time, Sample_ID, Transition`
   - **This file is the input for the filtering step**

### How to Run

```python
from scripts.CLAW_OzESI import Parse

parser = Parse(
    data_base_name_location="/path/to/Lipid_Database.xlsx",
    Project_Folder_data="/path/to/mzml/folder",
    Project_results="/path/to/results",
    file_name_to_save="parsed_results",
    tolerance=0.3,
    remove_std=True,
    save_data=True,
    batch_processing=True,
    plot_chromatogram=False
)

# This generates all three CSV files
df_full_matched, ozesi_time_df, df_OzESI_matched = parser.mrm_run_all(deuterated=False)
```

Or use the standalone script:
```bash
cd /home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA
python3 generate_df_OzESI_matched.py
```

---

## Step 2: Filter and Process df_OzESI_matched.csv

### Input File
- **`df_OzESI_matched.csv`** (225,359 rows)
  - Generated in Step 1
  - Contains retention time and intensity data for all time points

### Script: `CLAW_OzESI_filter.py`

**Key Class**: `OzESIFilter`

**Key Method**: `run_all()`

This method performs:
1. **Filter by RT and Intensity** - Remove data outside RT range (10-23 min) and below intensity threshold (100)
2. **Calculate DB Positions** - Calculate double bond positions (n-7, n-9, n-12) based on parent ions
3. **Match Lipid Info** - Use DB positions to identify additional lipid matches
4. **Calculate Intensity Ratios** - Compute ratios between n-9 and n-7 fragments
5. **Filter Highest Ratios** - Keep only best matches per lipid/sample group

### Output File
- **`filtered_results.csv`** or **`df5csv.csv`** (98 rows)
  - Filtered, processed data with DB positions and ratios
  - Columns: All original columns plus:
    - `n-7`, `n-9`, `n-12` (calculated DB position values)
    - `db_pos` (assigned DB position label)
    - `Ratio` (n-9/n-7 intensity ratio)
    - `is_special_case` (flag for special filtering cases)

### How to Run

```python
from scripts.CLAW_OzESI_filter import OzESIFilter

filter = OzESIFilter(tolerance=0.3)

message = filter.run_all(
    df_csv="/path/to/df_OzESI_matched.csv",  # Input from Step 1
    min_rt=10,
    max_rt=23,
    min_intensity=100,
    db_pos_list="7,9,12",
    sort_by_columns="Sample_ID,Product_Ion",
    tolerance=0.3,
    save_csv_path="/path/to/filtered_results.csv"  # Output
)
```

Or use the test script:
```bash
cd /home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA
python3 test_filter.py
```

---

## Complete Workflow Example

```python
# ==================== STEP 1: Parse mzML files ====================
from scripts.CLAW_OzESI import Parse

parser = Parse(
    data_base_name_location="/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/lipid_database/Lipid_Database.xlsx",
    Project_Folder_data="/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/Projects/canola/mzml/ON",
    Project_results="/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results",
    file_name_to_save="parsed_results",
    tolerance=0.3,
    remove_std=True,
    save_data=True,
    batch_processing=True,
    plot_chromatogram=False
)

# Generate df_OzESI_matched.csv (and other files)
df_full_matched, ozesi_time_df, df_OzESI_matched = parser.mrm_run_all()
print(f"Generated df_OzESI_matched.csv with {len(df_OzESI_matched)} rows")

# ==================== STEP 2: Filter the data ====================
from scripts.CLAW_OzESI_filter import OzESIFilter

filter = OzESIFilter(tolerance=0.3)

message = filter.run_all(
    df_csv="df_OzESI_matched.csv",  # File from Step 1
    min_rt=10,
    max_rt=23,
    min_intensity=100,
    db_pos_list="7,9,12",
    sort_by_columns="Sample_ID,Product_Ion",
    tolerance=0.3,
    save_csv_path="filtered_results.csv"
)
print(message)
```

---

## File Locations

### Source Scripts
```
/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/scripts/
├── CLAW_OzESI.py         - Parses mzML files and generates df_OzESI_matched.csv
└── CLAW_OzESI_filter.py  - Filters and processes df_OzESI_matched.csv
```

### Input Data
```
/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/lipid_platform/
├── lipid_database/
│   └── Lipid_Database.xlsx  - MRM reference database
└── Projects/canola/mzml/ON/
    ├── CrudeCanola_O3on_150gN3_02082023.mzML
    ├── DegummedCanola_O3on_150gN3_02082023.mzML
    └── RBDCanola_O3on_150gN3_02082023.mzML
```

### Output Data
```
/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results/
├── df_full_matched.csv      - Summary matched data (105 rows)
├── ozesi_time_df.csv        - Raw chromatogram data (225,357 rows)
├── df_OzESI_matched.csv     - Matched chromatogram data (225,359 rows) ⭐ Input for filter
└── filtered_results.csv     - Final filtered data (98 rows)
```

---

## Key Differences Between Files

### parsed_results.csv (From AI agent - 105 rows)
- ❌ **Missing `Retention_Time`** - Cannot be used for filtering
- ❌ Has `Intensity` instead of `OzESI_Intensity`
- ℹ️ This is the summarized version (df_full_matched.csv)

### df_OzESI_matched.csv (From mrm_run_all - 225,359 rows)
- ✅ **Has `Retention_Time`** - Required for filtering
- ✅ **Has `OzESI_Intensity`** - Correct column name
- ✅ Time-series data with all chromatogram points
- ⭐ **This is the correct input for CLAW_OzESI_filter.py**

---

## Why parsed_results.csv Doesn't Work

The `parsed_results.csv` in `agent_results/` is actually the output of `df_full_matched.csv`, which:
1. Is a **summary** with total intensity per transition (no time series)
2. **Lacks retention time data** (aggregated/lost during summarization)
3. Only has 105 rows (one per unique transition across all samples)

The filter script needs **chromatogram-level data** with retention times to:
- Filter by retention time (10-23 minutes)
- Analyze peak shapes and select optimal time points
- Calculate DB positions based on chromatographic behavior

---

## Quick Start Commands

### Generate df_OzESI_matched.csv
```bash
cd /home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA
python3 generate_df_OzESI_matched.py
```

### Filter the data
```bash
cd /home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA
python3 test_filter.py
```

---

## Summary

**Data Flow:**
1. **Raw mzML files** (mass spec data)
2. → `CLAW_OzESI.py::mrm_run_all()` →
3. **df_OzESI_matched.csv** (225,359 rows with retention times)
4. → `CLAW_OzESI_filter.py::run_all()` →
5. **filtered_results.csv** (98 rows with DB positions and ratios)

**Key Point:** You need to run `mrm_run_all()` from `CLAW_OzESI.py` to generate the proper `df_OzESI_matched.csv` file with retention time data before you can use `CLAW_OzESI_filter.py`.
