import glob
import numpy as np
import h5py
import cupy
import matplotlib.pyplot as plt

h5_files = sorted(glob.glob("output_job*.h5"))

if not h5_files:
    print("No output_job*.h5 files found.")
    raise SystemExit

all_prey     = []
all_predator = []

for path in h5_files:
    with h5py.File(path, "r") as f:
        all_prey.append(f["prey"][:])
        all_predator.append(f["predator"][:])

prey     = np.concatenate(all_prey)      # shape (R_total, L)
predator = np.concatenate(all_predator)

mean_prey     = prey.mean()
mean_predator = predator.mean()

print(f"Mean prey density     : {mean_prey:.6f}")
print(f"Mean predator density : {mean_predator:.6f}")

# First realization spatial profile
L    = prey.shape[1]
sites = np.arange(L)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

axes[0].plot(sites, prey[0], color="tab:green", label="prey (seed 0)")
axes[0].axhline(mean_prey, color="tab:green", linestyle="--", label=f"mean = {mean_prey:.4f}")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].set_title("Prey")

axes[1].plot(sites, predator[0], color="tab:red", label="predator (seed 0)")
axes[1].axhline(mean_predator, color="tab:red", linestyle="--", label=f"mean = {mean_predator:.4f}")
axes[1].set_ylabel("Density")
axes[1].set_xlabel("Site")
axes[1].legend()
axes[1].set_title("Predator")

plt.tight_layout()
plt.show()
