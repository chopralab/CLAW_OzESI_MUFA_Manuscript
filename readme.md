# CLAW-OzESI-MRM: Agentic OzESI-MRM Processing Pipeline

## Overview
Here we introduce a feature of CLAW-OzESI-MRM for identifying double bond locations in monounsaturated fatty acids (MUFAs) using LC-OzESI-MRM (Ozone-induced Electrospray Ionization Multiple Reaction Monitoring). This repository contains both the core OzESI CLAW processing pipeline and its AI agent integration located in the ASPIRE_LINQX submodule.

## Key Features
- High-throughput MUFA double bond localization 
- LLM-Driven Agentic data processing, analysis, and visualization 
- Integration with commercially available Agilent 6495C QQQ 

## AI Agent Capabilities
The ASPIRE_LINQX submodule provides three specialized AI agents:
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

# Install dependencies
conda env create -f requirements/mufaAI.yml
```

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
└── __pycache__
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