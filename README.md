# ADME-Tox prediction models
AI based ADME/Tox prediction models for early stage drug screening

## Introduction: ## 

This repository provides AI-based models to efficiently predict 12 important ADMET properties of drug candidates:

**Absorption:**
Caco-2 permeability and P-gp substrate

**Distribution:**
Blood-Brain Barrier permeability

**Metabolism:** 
CYP1A2, CYP2C9, CYP2C19, CYP2D6, and CYP2A4

**Excretion:**
Human liver microsomal stability (HLM), Mouse liver microsomal stability (MLM), and Rat microsomal stability (RLM)

**Toxicity:**
hERG inhibition

## Dependencies ##

- Python ≥ 3.9
- scikit-learn ≥ 1.26.4
- numpy == 11.5.0
- hpsklearn == 1.0.3
- hyperopt == 0.2.7
- xgboost == 2.0.3
- rdkit
- pandas

## Execution ##
**To run the prediction:**

$ python model.py --prediction --file_name [filename]

Note: For the prediction step, prepare a .csv file containing SMILES without bioclass (e.g., test_set.csv)

**To run the validation:**

$ python model.py --validation --file_name [filename]

Note: For the validation step, prepare a .csv file containing SMILES with bioclass (0 or 1) (e.g., valid_set.csv)

## Model Files ## 

To access the prediction model files in .pkl format, please refer to the "Tag --> v2.3.4" tab. Extract the folder "best_models" and place it in the working directory.

