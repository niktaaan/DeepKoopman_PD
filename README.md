# DeepKoopman_PD
Code for the paper : " Koopman-Based Linearization of Preparatory EEG Dynamics in Parkinson’s Disease During Galvanic Vestibular Stimulation""

This repository contains the code, data samples, and experiments for our paper:

> **Title:** _Linearized Koopman Embeddings Reveal EEG Effects of Galvanic Vestibular Stimulation During Motor Preparation in Parkinson's Disease_  
> **Authors:** Maryam Kia, Maryam S. Mirian, Saeed Soori, Saeed Saedi, Emad Arasteh, Mohamad Hosein Faramarzi, Abhijit Chinchani, Soojin Lee, Artur Luczak, Martin J. McKeown*
> 
> **Affiliation:** UBC Pacific Parkinson’s Research Centre  
> **Preprint/Paper:** [[link-to-arxiv-or-journal](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1566566/abstract)] 

---

## 🧠 Overview

This study applies a Deep Koopman Operator Learning framework to EEG recordings from Parkinson’s patients and healthy controls, with and without galvanic vestibular stimulation (GVS). Our method captures both spatial and temporal dynamics in a latent linear space, allowing interpretable analysis of non-invasive brain stimulation effects.
![image](https://github.com/user-attachments/assets/f54b794f-69d4-40b0-bd72-e3c7ae495acd)

---

## 📁 Repository Structure

```bash
deep-kooopman-eeg-gvs/
│
├── data/                 # Sample EEG and behavioral data
├── src/                  # Source code: preprocessing, models, utils
├── notebooks/            # Jupyter notebooks for reproducing results
├── figures/              # Figures from the paper
├── results/              # Output trajectories, spatial maps, statistics
├── paper/                # Manuscript and supplementary materials
├── scripts/              # Scripts to run full pipeline
├── requirements.txt      # Python dependencies
├── CITATION.cff          # Citation metadata
├── LICENSE
└── README.md             # You're here!

🚀 Getting Started

(Installation, setup, environment — start this section here)

---

📊 Reproducing Results

(Outline which notebooks/scripts to run and how)

---

📎 Citation

(Add citation instructions)

---


