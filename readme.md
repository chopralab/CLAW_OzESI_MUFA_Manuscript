# CLAW-OzESI-MRM: Agentic OzESI-MRM Processing Pipeline

## Overview
Here we introduce a feature of CLAW-OzESI-MRM for identifying double bond locations in monounsaturated fatty acids (MUFAs) using LC-OzESI-MRM (Ozone-induced Electrospray Ionization Multiple Reaction Monitoring). This repository contains both the core OzESI CLAW processing pipeline and its AI agent integration located in the ASPIRE_LINQX submodule.

## Key Features
- High-throughput MUFA double bond localization 
- LLM-Driven Agentic data processing, analysis, and visualization 
- Integration, explained in the paper, for the commercially available Agilent 6495C QQQ and equivalent Agilent systems

## AI Agent Capabilities
The ASPIRE_LINQX submodule (Now known as SciBORG) provides three specialized AI agents:
1. **Data Parser**: Transforms mzml data into organized Pandas DataFrames
2. **Double-bond Localization**: Automatically predicts possible MRM m/z values for user selected double-bond locations and then identifies them in each sample
3. **Visualization Agent**: Visualization of results

## Experimental Data
This repository includes OzESI-MRM data from canola oil analysis at three production stages:
- Crude
- Degummed
- RBD

## Installation
```bash
# Clone the repository
git clone https://github.com/chopralab/CLAW_OzESI_MUFA_Manuscript.git
cd CLAW_OzESI_MUFA_Manuscript

# Create the conda environment
conda env create -f requirements/mufaAI.yml

# Activate the environment
conda activate mufaAI
```

All examples and notebooks below assume you have activated the `mufaAI` environment.

## Using the CLAW_OzESI Tutorial (Canola Oil Example)

The main end‑to‑end example for the CLAW OzESI pipeline is provided in the notebook:

`lipid_platform/CLAW_OzESI_Tutorial.ipynb`

This tutorial walks through processing canola oil LC‑OzESI‑MRM data from the `Projects/canola` folder.

1. Open `lipid_platform/CLAW_OzESI_Tutorial.ipynb`.
2. Ensure the kernel is set to the `mufaAI` conda environment.
3. Run the cells from top to bottom to:
   - Load the canola OzESI‑MRM data from `lipid_platform/Projects/canola/mzml/`.
   - Process and analyze MUFA double‑bond localization.
   - Generate plots and results under `lipid_platform/Projects/canola/plots/` and `results/`.

## Using MUFA AI Agents (Jupyter Notebooks)

The MUFA AI Agents are provided as six Jupyter notebooks under:

`ASPIRE_LINQX/integrations/MUFA`

Workflow notebooks (end‑to‑end agent usage):

- `MUFA_AI_Agent_Parsing.ipynb`
- `MUFA_AI_Agent_OzESIFilter.ipynb`
- `MUFA_AI_Agent_Plot.ipynb`

Error‑handling / debugging examples:

- `MUFA_AI_Agent_Error_Example_1.ipynb`
- `MUFA_AI_Agent_Error_Example_2.ipynb`
- `MUFA_AI_Agent_Error_Example_3.ipynb`

To run any of these notebooks:

```bash
conda activate mufaAI
jupyter lab
```

In Jupyter Lab:

1. Navigate to `ASPIRE_LINQX/integrations/MUFA`.
2. Open one of the MUFA AI Agent notebooks listed above.
3. Select the `mufaAI` kernel.
4. Run the cells from top to bottom.

The first three notebooks demonstrate how to parse OzESI‑MRM data, apply the OzESI MUFA agent workflow, and generate plots. The three `Error_Example` notebooks illustrate how the agents handle errors such as incorrect names or help requests.

## Project Structure - CLAW-OzESI-MRM 

```
lipid_platform
├── lipid_database
├── Projects
│   └── canola
│       ├── mzml
│       │   ├── OFF
│       │   └── ON
│       ├── plots
│       │   ├── ratio
│       │   └── stats
│       └── results
```

## Project Structure - AI Agents
```
.
├── ASPIRE_LINQX/
│   └── integrations/
│       └── MUFA/
│           └── scripts/
├── lipid_platform/
│   ├── lipid_database/
│   └── Projects/
│       └── canola/
│           ├── mzml/
│           ├── plots/
│           └── results/
└── requirements/
```

## Contact
For questions on the code please contact Sanjay Iyer iyer95@purdue.edu 