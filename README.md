# DeepKoopman_PD

Research code accompanying **Koopman-Based Linearization of Preparatory EEG Dynamics in Parkinson’s Disease During Galvanic Vestibular Stimulation**.

[Read the paper](https://doi.org/10.3389/fnhum.2025.1566566)

## Overview

This study uses deep Koopman learning to model preparatory EEG dynamics in Parkinson’s disease and healthy controls. An autoencoder maps EEG signals into a latent representation, where a learned dynamical model supports prediction and feedback-control analysis.

The repository includes a sample healthy-control training configuration, preprocessed EEG sample data, and postprocessing scripts.

![Study overview](https://github.com/user-attachments/assets/aa75f129-d49a-4b9e-a5d7-3923587a8668)

## Repository contents

| Path | Purpose |
| --- | --- |
| `data/` | Sample EEG data split into training, validation, and test CSV files |
| `src/ShamHC.py` | Configuration and entry point for the sample healthy-control experiment |
| `src/training.py` | Training procedure and loss functions |
| `src/networkarch.py` | Autoencoder and latent-dynamics network definitions |
| `src/helperfns.py` | Utilities for training and saving experiment outputs |
| `src/Postprocessing/` | Python and MATLAB scripts for analysis, plotting, and control |
| `requirements.txt` | Python dependency list |

## Sample experiment

Run the training entry point from the repository root:

```bash
python src/ShamHC.py
