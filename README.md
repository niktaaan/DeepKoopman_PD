# DeepKoopman_PD
Code for the paper : " Koopman-Based Linearization of Preparatory EEG Dynamics in Parkinson’s Disease During Galvanic Vestibular Stimulation""

This repository contains the code, data samples, and experiments for our paper:

> **Title:** _Koopman-Based Linearization of Preparatory EEG Dynamics in Parkinson’s Disease During Galvanic Vestibular Stimulation_  
> **Authors:** Maryam Kia, Maryam S. Mirian, Saeed Soori, Saeed Saedi, Emad Arasteh, Mohamad Hosein Faramarzi, Abhijit Chinchani, Soojin Lee, Artur Luczak, Martin J. McKeown*
> 
> **Affiliation:** UBC Pacific Parkinson’s Research Centre  
> **Preprint/Paper:** [[link-to-arxiv-or-journal](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1566566/abstract)] 

---

## 🧠 Overview

This study applies a Deep Koopman Operator Learning framework to EEG recordings from Parkinson’s patients and healthy controls, with and without galvanic vestibular stimulation (GVS). Our method captures both spatial and temporal dynamics in a latent linear space, allowing interpretable analysis of non-invasive brain stimulation effects.
![Figure1](https://github.com/user-attachments/assets/aa75f129-d49a-4b9e-a5d7-3923587a8668)

---

## 📁 Repository Structure

```bash
deep-kooopman-eeg-gvs/
│
├── data/                 
│   └── sample_eeg/             # Example EEG data from HC subjects (train/val/test sets)
│
├── src/                  
│   └── training/               # Core model definition and training pipeline
│
├── src/postprocessing/        # Scripts for downstream analysis and figure generation
│
├── paper/                     
│   ├── manuscript.pdf          # Main manuscript 
│   └── supplementary/          # Supplementary figures, tables, or data 
│
├── requirements.txt           # List of Python dependencies (for pip install)
├── CITATION.cff               # Citation metadata (for Zenodo/GitHub citation integration)
├── LICENSE                    # License file (e.g. MIT)
└── README.md                  # You’re here!

**What’s Included**

Sample Data:
A subset of preprocessed EEG recordings from HC subjects, formatted for immediate use. These include separate files for training, validation, and test.

Training Code (src/training/):
Code to define and train the deep Koopman model on EEG data. 


Postprocessing Scripts (src/postprocessing/):
Scripts used after training to Apply the Koopman operator to test data, Reconstruct signals and compute prediction errors, Analyze learned latent dynamics, Generate figures for visualization and publication

---

📎 Citation

(Add citation instructions)

---


