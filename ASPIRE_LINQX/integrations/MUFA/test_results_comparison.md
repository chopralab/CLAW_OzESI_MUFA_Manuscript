# CLAW_OzESI_filter.py Test Results

## Test Summary

The `CLAW_OzESI_filter.py` script was tested with two different CSV files.

### ✅ Test 1: df_OzESI_matched.csv (SUCCESS)

**File**: `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results/df_OzESI_matched.csv`

**Result**: ✅ **Script runs successfully**

**Columns Present**:
- `Class`
- `Lipid`
- `OzESI_Intensity` ✅ (Required)
- `Parent_Ion`
- `Product_Ion`
- `Retention_Time` ✅ (Required)
- `Sample_ID`
- `Transition`

**Output**: Successfully created `filtered_results.csv` with 98 rows of filtered and processed data.

**Output Columns**:
- All input columns plus:
  - `n-7`, `n-9`, `n-12` (calculated DB positions)
  - `db_pos` (double bond position labels)
  - `Ratio` (intensity ratios between n-9 and n-7)
  - `is_special_case` (flag for special filtering cases)

---

### ❌ Test 2: parsed_results.csv (FAILED)

**File**: `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/agent_results/parsed_results.csv`

**Result**: ❌ **Script fails with KeyError**

**Columns Present**:
- `Class`
- `Intensity` ❌ (Should be `OzESI_Intensity`)
- `Lipid`
- `Parent_Ion`
- `Product_Ion`
- `Sample_ID`
- `Transition`

**Missing Columns**:
1. ❌ `Retention_Time` - **Critical** (used for filtering)
2. ❌ `OzESI_Intensity` - Column exists as `Intensity` but needs renaming

**Error**: 
```
KeyError: 'Retention_Time'
```

---

## Recommendations

### Option 1: Use df_OzESI_matched.csv (Recommended)
This file has all the required columns and works perfectly with the script.

### Option 2: Fix parsed_results.csv
To use `parsed_results.csv`, you would need to:

1. **Add Retention_Time column**: Merge or join with source data that contains retention times
2. **Rename column**: Change `Intensity` to `OzESI_Intensity`

Example transformation:
```python
import pandas as pd

df = pd.read_csv('parsed_results.csv')
# Option 1: If you have retention time data in another file
# df = df.merge(retention_time_df, on=['Sample_ID', 'Transition'])

# Option 2: If retention times are not available, you may need to reprocess raw data
# ...

# Rename column
df = df.rename(columns={'Intensity': 'OzESI_Intensity'})
df.to_csv('parsed_results_fixed.csv', index=False)
```

### Option 3: Check lipid_RT_map.csv
The file `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/results/lipid_RT_map.csv` might contain retention time mapping that could be merged with `parsed_results.csv`.

---

## Script Parameters Used

```python
ozesi_filter = OzESIFilter(tolerance=0.3)
message = ozesi_filter.run_all(
    df_csv="<input_file>",
    min_rt=10,
    max_rt=23,
    min_intensity=100,
    db_pos_list="7,9,12",
    sort_by_columns="Sample_ID,Product_Ion",
    tolerance=0.3,
    save_csv_path="<output_file>"
)
```

---

## Test Files

- Test script: `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/test_filter.py`
- Output: `/home/sanjay/github/MUFA_paper/CLAW_OzESI_MUFA_Manuscript/ASPIRE_LINQX/integrations/MUFA/agent_results/filtered_results.csv`
