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

### ⚙️ Setup

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

### 📁 Define Data Paths

```python
# Define your project directories
# Examples paths
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
ories
(Project_results / "ratio_area").mkdir(parents=True, exist_ok=True)
(Project_results / "ratio_intensity").mkdir(parents=True, exist_ok=True)
```

## Run MRM Parser
```python
parser = Parse(
    data_base_name_location=data_base_name_location,
    Project_Folder_data=Project_Folder_data,
    Project_results=Project_results,
    file_name_to_save=file_name_to_save,
    tolerance=tolerance,
    remove_std=True,
    save_data=False,
    batch_processing=True,
    plot_chromatogram=True
)
```

## Match Ion Labels
```python
lipid_mrm = pd.read_csv("Projects/canola_tutorial/results/lipid_mrm.csv")
df_canola_lipids = ion_label_parser(
    lipid_mrm=lipid_mrm,
    df_OzESI_matched=df_OzESI_matched,
    ion_tolerance=0.3,
    rt_tolerance=1.0,
    ion_labels=['n-7', 'n-9']
)
```

## Generate Ratio Plots
```python
SAMPLES = [
    {'name': 'Crude', 'sample_file': 'df_CrudeCanola_O3on_150gN3_02082023_summary.csv', 
     'manual_file': 'df_manual_crude.csv'},
    {'name': 'Degummed', 'sample_file': 'df_DegummedCanola_O3on_150gN3_02082023_summary.csv', 
     'manual_file': 'df_manual_degummed.csv'},
    {'name': 'RBD', 'sample_file': 'df_RBDCanola_O3on_150gN3_02082023_summary.csv', 
     'manual_file': 'df_manual_RBD.csv'}
]

for sample in SAMPLES:
    sample_csv = Project_results / sample['sample_file']
    manual_csv = Project_results / sample['manual_file']
    
    if sample_csv.exists() and manual_csv.exists():
        # Area ratios
        plot_n9_n7_ratios(
            sample_csv_path=str(sample_csv),
            manual_csv_path=str(manual_csv),
            sample_label=f'{sample["name"]} CLAW Area',
            manual_label=f'{sample["name"]} Manual Area',
            file_path=str(Project_results / "ratio_area" / f"{sample['name']}_area_ratios")
        )
        
        # Intensity ratios
        plot_n9_n7_ratios_intensity(
            sample_csv_path=str(sample_csv),
            manual_csv_path=str(manual_csv),
            sample_label=f'{sample["name"]} CLAW Intensity',
            manual_label=f'{sample["name"]} Manual Area',
            file_path=str(Project_results / "ratio_intensity" / f"{sample['name']}_intensity_ratios")
        )

print(f"✓ Area plots: {Project_results / 'ratio_area'}")
print(f"✓ Intensity plots: {Project_results / 'ratio_intensity'}")
```

## Output

- **Area-based ratios**: `results/ratio_area/`
- **Intensity-based ratios**: `results/ratio_intensity/`