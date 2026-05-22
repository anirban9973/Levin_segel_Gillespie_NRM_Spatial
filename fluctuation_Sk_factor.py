import os
import glob
import numpy as np
import h5py
import cupy as cp

# ============================================================
# Parameters — must match input.dat
# ============================================================

V      = 100.0
T_vals = [1, 32, 4096]

OUT_DIR = "fluctuations_and_spectrums"

# ============================================================
# Discover HDF5 files
# ============================================================

h5_files = sorted(glob.glob("output_job*.h5"))
if not h5_files:
    print("No output_job*.h5 files found.")
    raise SystemExit

print(f"Found {len(h5_files)} HDF5 file(s).")

# ============================================================
# Process each integration window
# ============================================================

for T in T_vals:

    lbl = str(T)
    print(f"\n{'='*56}")
    print(f"  T = {T}")
    print(f"{'='*56}")

    # ----------------------------------------------------------
    # Load data for this window across all jobs
    # ----------------------------------------------------------

    all_prey, all_pred = [], []
    for path in h5_files:
        with h5py.File(path, "r") as f:
            all_prey.append(f[f"prey_T{lbl}"][:])
            all_pred.append(f[f"predator_T{lbl}"][:])

    prey = np.concatenate(all_prey, axis=0)   # (R_total, L)
    pred = np.concatenate(all_pred, axis=0)

    R_total, L = prey.shape
    print(f"  Realizations : {R_total}")
    print(f"  L            : {L}")

    # ----------------------------------------------------------
    # Grand mean <N> — single scalar over all realizations
    # and all L sites
    # ----------------------------------------------------------

    mean_prey = float(np.mean(prey))
    mean_pred = float(np.mean(pred))
    print(f"  <N> prey     : {mean_prey:.6e}")
    print(f"  <N> predator : {mean_pred:.6e}")

    # ----------------------------------------------------------
    # Move to GPU
    # ----------------------------------------------------------

    prey_gpu = cp.asarray(prey, dtype=cp.float64)
    pred_gpu = cp.asarray(pred, dtype=cp.float64)

    # ----------------------------------------------------------
    # Fluctuation field
    #   xi_i = (N_i - <N>) / sqrt(T * V)
    # shape: (R_total, L)
    # ----------------------------------------------------------

    norm = np.sqrt(T * V)
    xi_prey = (prey_gpu - mean_prey) / norm
    xi_pred = (pred_gpu - mean_pred) / norm

    # ----------------------------------------------------------
    # Structure factor
    #   S(k) = (1/L) * < |FFT(xi)|^2 >_ensemble
    # ----------------------------------------------------------

    fft_prey = cp.fft.fft(xi_prey, axis=1)
    fft_pred = cp.fft.fft(xi_pred, axis=1)

    S_prey = cp.mean(cp.abs(fft_prey)**2, axis=0) / L   # shape (L,)
    S_pred = cp.mean(cp.abs(fft_pred)**2, axis=0) / L

    k_values = np.arange(L)

    # ----------------------------------------------------------
    # Spatial correlation
    #   C_r = (1/L) * sum_i < xi_i * xi_{(i+r) mod L} >
    #
    # Via circular correlation theorem:
    #   sum_i xi_i * xi_{(i+r) mod L} = IFFT(|FFT(xi)|^2)[r]
    # (numpy IFFT includes the 1/L factor, so dividing by L
    #  gives the correctly normalised c_r for each realization)
    # ----------------------------------------------------------

    corr_prey = cp.fft.ifft(cp.abs(fft_prey)**2, axis=1).real / L   # (R, L)
    corr_pred = cp.fft.ifft(cp.abs(fft_pred)**2, axis=1).real / L

    C_prey = cp.mean(corr_prey, axis=0)[:L // 2]   # r = 0 .. L/2-1
    C_pred = cp.mean(corr_pred, axis=0)[:L // 2]

    r_values = np.arange(L // 2)

    # ----------------------------------------------------------
    # Coarse-grained variance
    #   var(l) = < (1/l) * (1/L) * sum_r (sum_{i=r}^{r+l-1} xi_i)^2 >
    #
    # l = 1 .. L/2, periodic sliding windows via prefix sums
    # ----------------------------------------------------------

    l_values = np.arange(1, L // 2 + 1)
    var_prey = np.empty(len(l_values), dtype=np.float64)
    var_pred = np.empty(len(l_values), dtype=np.float64)

    print("  Computing coarse-grained variance...")
    for idx, l in enumerate(l_values):

        prey_ext = cp.concatenate([xi_prey, xi_prey[:, :l]], axis=1)
        pred_ext = cp.concatenate([xi_pred, xi_pred[:, :l]], axis=1)

        prey_cs = cp.cumsum(prey_ext, axis=1)
        pred_cs = cp.cumsum(pred_ext, axis=1)

        prey_win = prey_cs[:, l:] - prey_cs[:, :-l]   # (R, L) sliding sums
        pred_win = pred_cs[:, l:] - pred_cs[:, :-l]

        var_prey[idx] = float(cp.mean(cp.mean(prey_win**2, axis=1) / l))
        var_pred[idx] = float(cp.mean(cp.mean(pred_win**2, axis=1) / l))

        if idx % 20 == 0:
            print(f"    l = {l}")

    # ----------------------------------------------------------
    # Save outputs to fluctuations_and_spectrums/T{T}/
    # ----------------------------------------------------------

    out_dir = os.path.join(OUT_DIR, f"T{lbl}")
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(
        os.path.join(out_dir, "structure_factor_prey.dat"),
        np.column_stack([k_values, cp.asnumpy(S_prey.real)]),
        header="k S_k_prey"
    )
    np.savetxt(
        os.path.join(out_dir, "structure_factor_predator.dat"),
        np.column_stack([k_values, cp.asnumpy(S_pred.real)]),
        header="k S_k_predator"
    )

    np.savetxt(
        os.path.join(out_dir, "correlation_prey.dat"),
        np.column_stack([r_values, cp.asnumpy(C_prey)]),
        header="r C_r_prey"
    )
    np.savetxt(
        os.path.join(out_dir, "correlation_predator.dat"),
        np.column_stack([r_values, cp.asnumpy(C_pred)]),
        header="r C_r_predator"
    )

    np.savetxt(
        os.path.join(out_dir, "fluctuation_prey.dat"),
        np.column_stack([l_values, var_prey]),
        header="l var_l_prey"
    )
    np.savetxt(
        os.path.join(out_dir, "fluctuation_predator.dat"),
        np.column_stack([l_values, var_pred]),
        header="l var_l_predator"
    )

    np.savetxt(
        os.path.join(out_dir, "grand_mean.dat"),
        np.array([[mean_prey, mean_pred]]),
        header="mean_N_prey mean_N_predator"
    )

    print(f"  Saved to {out_dir}/")

print("\nAll done.")
