# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:31:55 2025

@author: Nikta
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os

# Load test error data from a JSON file
with open(".results/test_errors.json") as f:
    data = json.load(f)

# Define experimental groupings
grouped_data = {
    "HC": ["ShamHC"],
    "PD-Off": ["ShamPD-Off", "GVS1PD-Off", "GVS2PD-Off"],
    "PD-On": ["ShamPD-On", "GVS1PD-On", "GVS2PD-On"]
}

# Assign boxplot x-positions
positions = []
current_pos = 1
for group_conditions in grouped_data.values():
    positions.extend(range(current_pos, current_pos + len(group_conditions)))
    current_pos += len(group_conditions) + 1

# Flatten data for plotting
flattened_data = [data[group] for group_list in grouped_data.values() for group in group_list]

# Box color assignment
colors = {"Sham": "red", "GVS1": "blue", "GVS2": "green"}
box_colors = [
    colors[cond.split("PD-")[0] if "PD-" in cond else "Sham"]
    for cond in data.keys()
]

# Plot
plt.figure(figsize=(12, 12))
bp = plt.boxplot(
    flattened_data,
    positions=positions,
    patch_artist=True,
    showmeans=True,
    meanline=True,
    medianprops={"visible": False},
    meanprops={"color": "black", "linestyle": "--"}
)

for box, color in zip(bp['boxes'], box_colors):
    box.set_facecolor(color)

# Group labels centered under each section
group_labels = list(grouped_data.keys())
group_positions = [
    np.mean(positions[i:i + len(v)])
    for i, v in zip(range(0, len(positions), 4), grouped_data.values())
]
plt.xticks(group_positions, group_labels, fontsize=12)

# Axis labels and grid
plt.xlabel("Experimental Groups", fontsize=12)
plt.ylabel("Network Error (Test Set)", fontsize=12)
plt.grid(linestyle='--', alpha=0.7)

# Legend
legend_elements = [
    Line2D([0], [0], color="black", linestyle="--", lw=2, label="Mean (line)"),
    Line2D([0], [0], color="red", lw=4, label="Sham"),
    Line2D([0], [0], color="blue", lw=4, label="GVS1"),
    Line2D([0], [0], color="green", lw=4, label="GVS2")
]
plt.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=False)

# Save to multiple formats
os.makedirs("figures", exist_ok=True)
for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure1a.{fmt}", format=fmt, dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# =============================================================================
# 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os

# Load test data and reconstructions
traj_data = np.load("results/traj_data_test.npy")           # shape (N, 5)
traj_recon = np.load("results/traj_data_recon.npy")         # shape (N, 5)

# Use first 10000 samples
traj_data = traj_data[:10000, :]
traj_recon = traj_recon[:10000, :]
time = np.arange(traj_data.shape[0])  # in milliseconds

# === Figure 1B: Original vs Reconstructed ===
fig, axes = plt.subplots(5, 1, figsize=(12, 6), sharex=True, sharey=True)
subtitles = ["x1", "x2", "x3", "x4", "x5"]

for i, ax in enumerate(axes):
    ax.plot(time, traj_data[:, i], label="Original", alpha=0.7, linewidth=1.5)
    ax.plot(time, traj_recon[:, i], label="Reconstructed", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(subtitles[i], fontsize=10)
    if i == len(axes) - 1:
        ax.set_xlabel("Time (ms)", fontsize=10)

fig.text(0.04, 0.5, "Amplitude", va="center", ha="center", rotation="vertical", fontsize=10)
axes[0].legend(loc="upper right", fontsize=9)
plt.tight_layout(rect=[0.05, 0, 1, 1])

os.makedirs("figures", exist_ok=True)
for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure1b.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
plt.show()

# === Figure 1C: Zoomed-in view (x1 only) ===
zoom_start, zoom_end = 4000, 6000
time_zoom = np.arange(zoom_start, zoom_end)

plt.figure(figsize=(10, 4))
plt.plot(time_zoom, traj_data[zoom_start:zoom_end, 0], label="Original", alpha=0.7, linewidth=1.5)
plt.plot(time_zoom, traj_recon[zoom_start:zoom_end, 0], label="Reconstructed", linestyle="--", linewidth=1.5)
plt.xlabel("Time (ms)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", fontsize=9, frameon=False)
plt.tight_layout()

for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure1c.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Prediction Error
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os

# === FIGURE 2A: Boxplot of 10-step prediction errors ===

with open("results/figure2a/prediction_error_10step_map.json") as f:
    file_map = json.load(f)

data = {
    group: np.load(f"results/figure2a/{filename}").tolist()
    for group, filename in file_map.items()
}

grouped_data = {
    "HC": ["ShamHC"],
    "PD-Off": ["ShamPD-Off", "GVS1PD-Off", "GVS2PD-Off"],
    "PD-On": ["ShamPD-On", "GVS1PD-On", "GVS2PD-On"]
}

positions, current_pos = [], 1
for group in grouped_data.values():
    positions.extend(range(current_pos, current_pos + len(group)))
    current_pos += len(group) + 1

flattened_data = [data[cond] for group in grouped_data.values() for cond in group]
colors = {"Sham": "red", "GVS1": "blue", "GVS2": "green"}
box_colors = [colors[cond.split("PD-")[0] if "PD-" in cond else "Sham"] for cond in file_map]

plt.figure(figsize=(12, 12))
bp = plt.boxplot(flattened_data, positions=positions, patch_artist=True,
                 showmeans=True, meanline=True,
                 medianprops={"visible": False},
                 meanprops={"color": "black", "linestyle": "--"})
for box, color in zip(bp['boxes'], box_colors):
    box.set_facecolor(color)

plt.xticks([1, 4, 8], grouped_data.keys(), fontsize=12)
plt.xlabel("Experimental Groups", fontsize=12)
plt.ylabel("Normalized 10-Step Prediction Error", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

legend_elements = [
    Line2D([0], [0], color="black", linestyle="--", lw=2, label="Mean (line)"),
    Line2D([0], [0], color="red", lw=4, label="Sham"),
    Line2D([0], [0], color="blue", lw=4, label="GVS1"),
    Line2D([0], [0], color="green", lw=4, label="GVS2")
]
plt.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=False)

os.makedirs("figures", exist_ok=True)
for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure2a.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# === FIGURE 2B: Stepwise prediction errors across 15 steps ===

with open("results/figure2b/prediction_error_stepwise_map.json") as f:
    stepwise_map = json.load(f)
with open("results/figure2b/normalization_factors.json") as f:
    norm_factors = json.load(f)

stepwise_data = {
    group: np.load(f"results/figure2b/{filename}")[:, :15] / norm_factors[filename.split('_')[0]]
    for group, filename in stepwise_map.items()
}

plt.figure(figsize=(12, 8))
steps = np.arange(1, 16)

for group, values in stepwise_data.items():
    mean = np.mean(values, axis=0)
    sem = np.std(values, axis=0) / np.sqrt(values.shape[0])
    plt.plot(steps, mean, label=group, linewidth=2)
    plt.fill_between(steps, mean - sem, mean + sem, alpha=0.06)

plt.xlabel("Prediction Step", fontsize=12)
plt.ylabel("Normalized Prediction Error", fontsize=12)
plt.grid(linestyle="--", alpha=0.6)
plt.legend(fontsize=10, loc="upper left")
plt.tight_layout()

for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure2b.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Eigenvalues
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os

os.makedirs("figures", exist_ok=True)

# === FIGURES 3A and 3B: Eigenvalue scatter plots ===

with open("results/figure3/eigenvalues_real.json") as f:
    real_vals = np.array(json.load(f))

with open("results/figure3/eigenvalues_complex.json") as f:
    complex_vals = np.array(json.load(f)).view(np.complex128)

unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 500))

def save_plot(fig_id):
    for fmt in ["eps", "pdf", "tiff", "png"]:
        plt.savefig(f"figures/figure3{fig_id}.{fmt}", format=fmt, dpi=300, bbox_inches='tight')

# -- 3A: Full eigenvalue plot
plt.figure(figsize=(10, 10))
plt.scatter(real_vals, np.zeros_like(real_vals), color="blue", label="Real Eigenvalues", facecolors='none')
plt.scatter(complex_vals.real, complex_vals.imag, color="red", label="Complex Eigenvalues", facecolors='none')
plt.plot(unit_circle.real, unit_circle.imag, linestyle="--", color="black", label="Unit Circle")
plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(alpha=0.3)
plt.axis("equal")
plt.legend()
save_plot("a")
plt.show()

# -- 3B: Zoomed-in view
plt.figure(figsize=(8, 8))
plt.scatter(real_vals, np.zeros_like(real_vals), color="blue", label="Real Eigenvalues", facecolors='none')
plt.scatter(complex_vals.real, complex_vals.imag, color="red", label="Complex Eigenvalues", facecolors='none')
plt.plot(unit_circle.real, unit_circle.imag, linestyle="--", color="black", label="Unit Circle")
plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
plt.xlim(0.92, 1.01)
plt.ylim(-0.015, 0.015)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(alpha=0.3)
plt.legend()
save_plot("b")
plt.show()

# === FIGURES 3C, 3D, 3E: Boxplots of Eigenvalue Components ===

def plot_boxplot(data_dict, ylabel, fig_id):
    colors = ["red", "red", "blue", "green", "blue", "green", "green"]
    positions = [1, 3, 4, 5, 7, 8, 9]
    group_labels = ["HC", "PD-Off", "PD-On"]

    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(data_dict.values(), positions=positions, patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops={"color": "black", "linestyle": "--"},
                     medianprops={"visible": False})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks([1, 4, 8], group_labels)
    plt.xlabel("Experimental Groups")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    legend = [
        Line2D([0], [0], color="red", lw=4, label="Sham"),
        Line2D([0], [0], color="blue", lw=4, label="GVS1"),
        Line2D([0], [0], color="green", lw=4, label="GVS2"),
        Line2D([0], [0], color="black", linestyle="--", label="Mean (line)")
    ]
    plt.legend(handles=legend, loc="best", fontsize=10, frameon=False)
    plt.tight_layout()
    save_plot(fig_id)
    plt.show()

with open("results/figure3/real_eigenvalue_by_group.json") as f:
    real_by_group = {k: np.array(v) for k, v in json.load(f).items()}
plot_boxplot(real_by_group, "Real Eigenvalues", "c")

with open("results/figure3/real_part_complex_by_group.json") as f:
    realpart_by_group = {k: np.array(v) for k, v in json.load(f).items()}
plot_boxplot(realpart_by_group, "Real Part of Complex Eigenvalues", "d")

with open("results/figure3/imag_part_complex_by_group.json") as f:
    imagpart_by_group = {k: np.array(v) for k, v in json.load(f).items()}
plot_boxplot(imagpart_by_group, "Imaginary Part of Complex Eigenvalues", "e")


# =============================================================================
# Distance from baseline(from coef_distance.m )
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os

# Load JSON files for each dimension
with open("results/figure6/distance_dim1.json") as f:
    data_dim1 = json.load(f)
with open("results/figure6/distance_dim2.json") as f:
    data_dim2 = json.load(f)
with open("results/figure6/distance_dim3.json") as f:
    data_dim3 = json.load(f)

# Shared structure
grouped_conditions = {
    "PD-Off": ["ShamPD-Off", "GVS1PD-Off", "GVS2PD-Off"],
    "PD-On": ["ShamPD-On", "GVS1PD-On", "GVS2PD-On"]
}
colors = {"Sham": "red", "GVS1": "blue", "GVS2": "green"}

# --- BOXPLOT (Dim1 only as example for figure6) ---
positions = []
current_pos = 1
for conditions in grouped_conditions.values():
    positions.extend(range(current_pos, current_pos + len(conditions)))
    current_pos += len(conditions) + 1

flattened_data = [data_dim1[c] for g in grouped_conditions.values() for c in g]
box_colors = [colors[c.split("PD-")[0]] for c in data_dim1]

plt.figure(figsize=(12, 12))
bp = plt.boxplot(flattened_data, positions=positions, patch_artist=True, showmeans=True, meanline=True,
                 medianprops={"visible": False}, meanprops={"color": "black", "linestyle": "--"})
for box, color in zip(bp['boxes'], box_colors):
    box.set_facecolor(color)

plt.xticks([2, 6], grouped_conditions.keys(), fontsize=12)
plt.xlabel("Experimental Groups", fontsize=12)
plt.ylabel("Distance from the baseline", fontsize=12)
plt.grid(linestyle='--', alpha=0.7)

legend_elements = [
    Line2D([0], [0], color="black", linestyle="--", lw=2, label="Mean (line)"),
    Line2D([0], [0], color="red", lw=4, label="Sham"),
    Line2D([0], [0], color="blue", lw=4, label="GVS1"),
    Line2D([0], [0], color="green", lw=4, label="GVS2")
]
plt.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=False)
plt.tight_layout()
for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure6_boxplot.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
plt.show()

# --- BAR PLOT 

def calculate_mean_sem(values):
    arr = np.array(values)
    return arr.mean(), arr.std() / np.sqrt(len(arr))

dim1_data = data_dim1
means, sems, conditions, bar_colors = [], [], [], []

for group, conds in grouped_conditions.items():
    for cond in conds:
        mean, sem = calculate_mean_sem(dim1_data[cond])
        means.append(mean)
        sems.append(sem)
        conditions.append(cond.split("PD-")[0])
        bar_colors.append(colors[cond.split("PD-")[0]])

bar_width = 0.25
num_conds = 3
group_positions = np.arange(len(grouped_conditions)) * (num_conds * bar_width + 0.5)
x = [group + i * bar_width for group in group_positions for i in range(num_conds)]

plt.figure(figsize=(10, 6))
for i, stim in enumerate(["Sham", "GVS1", "GVS2"]):
    indices = [j for j in range(len(x)) if conditions[j] == stim]
    stim_means = [means[j] for j in indices]
    stim_sems = [sems[j] for j in indices]
    plt.bar([x[j] for j in indices], stim_means, bar_width, yerr=stim_sems,
            label=stim, color=colors[stim], capsize=5)

plt.xticks(group_positions + bar_width, grouped_conditions.keys(), fontsize=12)
plt.xlabel("Experimental Groups", fontsize=12)
plt.ylabel("Distance from Baseline", fontsize=12)
plt.legend(fontsize=10)
plt.grid(linestyle="--", alpha=0.6)
plt.tight_layout()

for fmt in ["eps", "pdf", "tiff", "png"]:
    plt.savefig(f"figures/figure6_barplot.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
plt.show()
