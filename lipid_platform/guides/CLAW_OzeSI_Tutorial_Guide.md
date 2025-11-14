No worries — since I can’t directly open `.ipynb` files, I can still create the guide based on what you told me (`CLAW_OzESI` + `peak_analysis` workflow).

Here’s a structured **Markdown guide** you can save alongside your notebook:

---

## 🧪 CLAW_OzESI Peak Analysis Tutorial

### 📘 Overview

This notebook demonstrates how to perform automated lipid peak analysis and ratio comparison for canola oil samples using the **CLAW_OzESI** workflow.
It integrates:

* Data parsing and cleaning
* Automated peak detection
* n-7 vs n-9 lipid ratio analysis
* Optional visualization with the `plot_TG` module

---

### ⚙️ 1. Setup

```python
# Import libraries
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import CLAW_OzESI
import plot_TG
import importlib

# Reload plotting module if edited
importlib.reload(plot_TG)
```

---

### 📁 2. Define Data Paths

```python
# Define your project directories
data_base_name_location = "/scratch/negishi/iyer95/iyer95/CLAW_OzESI_Paper/CLAW/lipid_platform/lipid_database"
analysis_dir = "/scratch/negishi/iyer95/iyer95/CLAW_OzESI_Paper/CLAW/analysis"
Project_results = f"{analysis_dir}/results"
```

Create subdirectories automatically:

```python
import os
os.makedirs(f"{Project_results}/ratio_area", exist_ok=True)
os.makedirs(f"{Project_results}/ratio_intensity", exist_ok=True)
```

---

### 🧬 3. Load Lipid Data

```python
# Load lipid dataframe
df_canola_lipids = pd.read_csv("df_Canola_Lipids.csv")

# Inspect columns
df_canola_lipids.head()
```

Expected columns:

```
Sample_ID | Lipid | Peak_Area | RT | Class | Intensity | ...
```

---

### 🧩 4. Configure Sample Groups

```python
SAMPLES = [
    {"name": "Crude", "sample_file": "df_CrudeCanola_O3on_150gN3_02082023_summary.csv", "manual_file": "df_manual_crude.csv"},
    {"name": "Degummed", "sample_file": "df_DegummedCanola_O3on_150gN3_02082023_summary.csv", "manual_file": "df_manual_degummed.csv"},
    {"name": "RBD", "sample_file": "df_RBDCanola_O3on_150gN3_02082023_summary.csv", "manual_file": "df_manual_RBD.csv"}
]
```

---

### 🎛️ 5. Run Peak Analysis

```python
SHOW_PLOTS = False  # set True to show interactive figures

summary_df = CLAW_OzESI.peak_analysis(
    df_canola_lipids,
    output_dir=str(analysis_dir),
    show_plots=SHOW_PLOTS,
    plot_TG_module=plot_TG
)
summary_df.head()
```

If you see:

```
ℹ️ No summaries were generated (no patterns found for any samples)
```

Check that:

* `Sample_ID` values match your configuration.
* `Lipid` labels include `n-7` / `n-9` (not `n7` / `n9`).
* The DataFrame has valid numeric intensities.

---

### 🧹 6. Troubleshooting

**Fix naming and whitespace**

```python
df_canola_lipids['Sample_ID'] = df_canola_lipids['Sample_ID'].astype(str).str.strip()
df_canola_lipids['Lipid'] = (
    df_canola_lipids['Lipid'].astype(str)
    .str.strip()
    .str.replace('_', ' ')
    .str.replace('n7', 'n-7')
    .str.replace('n9', 'n-9')
)
```

**Verify sample coverage**

```python
df_canola_lipids['Sample_ID'].unique()
df_canola_lipids['Lipid'].unique()[:10]
```

---

### 📊 7. Generate Ratio Plots

```python
# Example: n-9 vs n-7 comparison
plot_TG.plot_n9_n7_ratios(summary_df, output_dir=Project_results)
```

---

### 📈 8. Save Results

```python
summary_df.to_csv(f"{Project_results}/summary_results.csv", index=False)
```

---

### 🧠 9. Notes

* This workflow is modular: you can reuse the `CLAW_OzESI.peak_analysis()` function for other lipid classes by adjusting lipid naming patterns.
* If you have a custom configuration, generate it via:

  ```python
  config = plot_TG.create_custom_config()
  CLAW_OzESI.peak_analysis(df_canola_lipids, config=config)
  ```

---

### 🗂️ Output Files

| Folder                | Description                  |
| --------------------- | ---------------------------- |
| `ratio_area/`         | Area-based lipid ratios      |
| `ratio_intensity/`    | Intensity-based lipid ratios |
| `summary_results.csv` | Combined output summary      |

---

### ✅ End of Tutorial

This completes the CLAW_OzESI lipid peak analysis pipeline setup and execution.

---

