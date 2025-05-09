#!/usr/bin/env python3
"""Quick post‑processing script for a single Deep‑Koopman run.

Edit **RUN_ID**, **ROOT**, or **MAX_EPOCH** below if needed
"""

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import notebookfns as n  # helper functions supplied with the project

# ---------------------------------------------------------------------
# CONFIGURATION (change these three lines only)
RUN_ID: str = "shamhc_2"   # params['data_name'] / folder name inside outputs
ROOT: Path = Path(".")      # directory containing the output files
MAX_EPOCH: int = 16         # epochs to show in the error plot
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Locate the files produced during training
model_file = ROOT / f"{RUN_ID}_model.pkl"
error_file = model_file.with_name(model_file.name.replace("model.pkl", "error.csv"))
fname = str(model_file)  # original helper routines expect a string called *fname*

# ---------------------------------------------------------------------
# Plot training / validation error history
errors = np.loadtxt(error_file, delimiter=",")
n.PlotErrors(errors, range(MAX_EPOCH))
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Load parameter dictionary and print a concise summary
with model_file.open("rb") as f:
    params: dict = pickle.load(f, encoding="latin1")

print(f"validation error:            {params['minTest']:.2E}")
print(f"training files:              {params['data_train_len']}")
print(f"trajectory length (steps):   {params['len_time']}")
print(f"batch size:                  {params['batch_size']}")
print(f"delta_t (s):                 {params['delta_t']:.3f}")
T = params['delta_t'] * (params['len_time'] - 1)
print(f"time span:                   0 – {T:.3f} s\n")

print("--- Loss weights (log10) ---")
print(f"alpha_1 (reconstruction)     {np.log10(params['recon_lam']):.1f}")
print(f"alpha_2 (‖·‖∞ term)          {np.log10(params['Linf_lam']):.1f}")
print(f"alpha_3 (L₂ reg.)            {np.log10(params['L2_lam']):.1f}\n")

print("--- Network architecture ---")
print(f"autoencoder pre-training?     {bool(params['auto_first'])}")
print(f"learning rate                {params['learning_rate']:.2E}")
depth = (params['d'] - 4) // 2
print(f"hidden depth (enc/dec)        {depth} layers")
print(f"layer widths (main)           {params['widths']}")
print(f"aux-net hidden layers         {len(params['hidden_widths_omega'])}")
print(f"aux-net widths                {params['hidden_widths_omega']}\n")

# ---------------------------------------------------------------------

print(f"We penalised {params['num_shifts']} (S_p) steps for prediction.")
print(f"We penalised {params['num_shifts_middle']} steps in the linearity loss.\n")

# Load all weights & biases
W, b = n.load_weights_koopman(
    fname,
    len(params['widths']) - 1,
    len(params['widths_omega_real']) - 1,
    params['num_real'],
    params['num_complex_pairs'],
)

# =============================================================================
# # Test‑set evaluation-save network test error 
# =============================================================================
params['data_name'] = RUN_ID  

test_x_file = ROOT / f"{params['data_name']}_test_x.csv"
X = np.loadtxt(test_x_file, delimiter=",")

# Reshape the data into a stacked form expected by helper routines
max_shifts_to_stack = n.num_shifts_in_stack(params)
X_stacked, num_traj_val = n.stack_data(X, max_shifts_to_stack, params['len_time'])
print(f"We used {num_traj_val} trajectories in the Test set.\n")

# Initial conditions
Xk = np.squeeze(X_stacked[0, :, :])

# Apply network to initial conditions
(
    yk,
    ykplus1,
    ykplus2,
    ykplus3,
    xk_recon,
    xkplus1,
    xkplus2,
    xkplus3,
) = n.ApplyKoopmanNetOmegas(
    Xk,
    W,
    b,
    params['delta_t'],
    params['num_real'],
    params['num_complex_pairs'],
    params['num_encoder_weights'],
    params['num_omega_weights'],
    params['num_decoder_weights'],
)

# Apply network to the full stacked dataset

y, g_list = n.ApplyKoopmanNetOmegasFull(
    X_stacked,
    W,
    b,
    params['delta_t'],
    params['num_shifts'],
    params['num_shifts_middle'],
    params['num_real'],
    params['num_complex_pairs'],
    params['num_encoder_weights'],
    params['num_omega_weights'],
    params['num_decoder_weights'],
)

# Compute loss components
(
    loss1_val,
    loss2_val,
    loss3_val,
    loss_Linf_val,
    loss_val,
) = n.define_loss(X_stacked, y, g_list, params, W, b)

print(f"Reconstruction loss (val):        {loss1_val:.4E}")
print(f"Prediction loss (val):            {loss2_val:.4E}")
print(f"Linearity loss (val):             {loss3_val:.4E}")
print(f"L_inf loss (val):                 {loss_Linf_val:.4E}")
print(f"Pre‑regularised loss (Table 1):   {loss_val:.4E}\n")

loss_L1_val, loss_L2_val, regularized_loss_val = n.define_regularization(params, W, b, loss_val)
print(f"L1 penalty (weights):             {loss_L1_val:.4E}")
print(f"L2 penalty (weights):             {loss_L2_val:.4E}")
print(f"Total regularised loss:           {regularized_loss_val:.4E}\n")

print("Sanity check against training record:")
print(f"  min validation loss during training:     {params['minTest']:.4E}")
print(f"  min regularised validation loss:         {params['minRegTest']:.4E}\n")
# =============================================================================
# passing computed u from decoder and finding u_original
# =============================================================================

x=Xk

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def encoder_apply(x, weights, biases, name, num_weights, act_type='relu'):
    prev_layer = x.copy()

    for i in np.arange(num_weights - 1):
        h1 = np.dot(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]

        if act_type == 'sigmoid':
            h1 = sigmoid(h1)
        elif act_type == 'relu':
            h1 = relu(h1)

        prev_layer = h1.copy()
    final = np.dot(prev_layer, weights['W%s%d' % (name, num_weights)]) + biases['b%s%d' % (name, num_weights)]

    return final


def decoder_apply(prev_layer, weights, biases, name, num_weights, act_type='relu'):
    for i in np.arange(num_weights - 1):
        h1 = np.dot(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]

        if act_type == 'sigmoid':
            h1 = sigmoid(h1)
        elif act_type == 'relu':
            h1 = relu(h1)
        prev_layer = h1.copy()
    final = np.dot(prev_layer, weights['W%s%d' % (name, num_weights)]) + biases['b%s%d' % (name, num_weights)]

    return final


yk = encoder_apply(x, W, b, 'E', params['num_encoder_weights'])
xk_recon = decoder_apply(yk, W, b, 'D', params['num_decoder_weights'])

# Load the CSV file
u = np.loadtxt('u_LQR.csv', delimiter=',')
u=u.T
u_original=decoder_apply(u, W, b, 'D', params['num_decoder_weights'])
# Save the array to a CSV file
np.savetxt('u_original.csv', u_original, delimiter=',')


# =============================================================================
# Trajectory‑level analysis - save encoded data
# =============================================================================
TRAJ_SHAPE = (30, 1000, 5)  # (n_traj, timesteps, channels)
traj_data = X.reshape(TRAJ_SHAPE)

omegas = np.zeros((30, 1000, 3))
traj_omega_mu_av = np.zeros((30, 3))  # per‑trajectory mean of [omega, mu1, mu2]

traj_yk = np.zeros((30, 1000, 3))
traj_recon = np.zeros_like(traj_data)
traj_pred1 = np.zeros_like(traj_data)
traj_pred3 = np.zeros_like(traj_data)

for j in range(traj_data.shape[0]):
    single_traj = traj_data[j]

    # Encode + predict
    (
        yk_traj,
        _,
        _,
        ykplus3_traj,
        xk_recon_traj,
        xkplus1_traj,
        _,
        xkplus3_traj,
    ) = n.ApplyKoopmanNetOmegas(
        single_traj,
        W,
        b,
        params['delta_t'],
        params['num_real'],
        params['num_complex_pairs'],
        params['num_encoder_weights'],
        params['num_omega_weights'],
        params['num_decoder_weights'],
    )

    # Auxiliary net → omegas & mus (time × 3)
    omega_out = n.omega_net_apply(
        yk_traj,
        W,
        b,
        params['num_real'],
        params['num_complex_pairs'],
        params['num_omega_weights'],
    )
    omegas[j, :, :2] = np.asarray(omega_out[0])
    omegas[j, :, 2] = np.asarray(omega_out[1]).ravel()
    traj_omega_mu_av[j] = omegas[j].mean(axis=0)

    # Store encoded / reconstructed / predicted signals
    traj_yk[j] = yk_traj
    traj_recon[j] = xk_recon_traj
    traj_pred1[j] = xkplus1_traj
    traj_pred3[j] = xkplus3_traj

omega_av, mu1_av, mu2_av = traj_omega_mu_av.T
print(f"Omega ranges from {omega_av.min():.7f} to {omega_av.max():.7f}")
print(f"Mu1 ranges from  {mu1_av.min():.7f} to {mu1_av.max():.7f}")
print(f"Mu2 ranges from  {mu2_av.min():.7f} to {mu2_av.max():.7f}\n")


# =============================================================================
# # Approximate K matrix and eigenvalue components
# =============================================================================
omg_mean = float(omega_av.mean())
mu1_mean = float(mu1_av.mean())
mu2_mean = float(mu2_av.mean())

dt = params['delta_t']
scale = np.exp(mu1_mean * dt)
entry11 = scale * np.cos(omg_mean * dt)
entry12 = scale * np.sin(omg_mean * dt)

L_approx = np.array([[entry11, -entry12], [entry12, entry11]])

evals_net, _ = np.linalg.eig(L_approx)
exp_mu2_dt = np.exp(mu2_mean * dt)

print("Approx. eigenvalues of learned K matrix:")
print("  from omega/mu1: ", evals_net)
print(f"  from mu2 only:  {exp_mu2_dt:.6f}")
