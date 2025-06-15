import numpy as np
import matplotlib.pyplot as plt
import os

# Apply academic-style Matplotlib settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 0.9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "figure.dpi": 300,
    "savefig.transparent": True,
    "grid.alpha": 0.4,
    "text.usetex": False,
})

# Simulation function for module forces
def simulate_module_forces(wave_amplitude, wave_period, a, b, n_modules, beam_length, gap_distance, total_time=10, dt=0.01):
    t = np.arange(0, total_time, dt)
    omega = 2 * np.pi / wave_period
    g = 9.81
    k = omega**2 / g
    pitch = beam_length + gap_distance
    x_positions = np.array([i * pitch for i in range(n_modules)])
    F_modules = np.zeros((n_modules, len(t)))
    
    for i, x in enumerate(x_positions):
        F_modules[i, :] = a * np.sin(k * x - omega * t) + b
        
    F_total = np.sum(F_modules, axis=0)
    return t, F_total, F_modules, x_positions

# Parameters
wave_amplitude = 1.0
wave_period = 5.0
F_avg = 3.04
F_max = 40.34
a = F_max - F_avg  # Oscillatory force amplitude
b = F_avg  # Constant drift force component
n_modules = 400
beam_length = 1.3
gap_distance = 0.7
total_time = 5* wave_period  # Simulate for 5 wave periods
dt = 0.01

# Run simulation
t, F_total, F_modules, x_positions = simulate_module_forces(
    wave_amplitude, wave_period, a, b, n_modules, beam_length, gap_distance, total_time, dt
)

# Plot total force on system
os.makedirs("plots", exist_ok=True)
fig, ax = plt.subplots(figsize=(5.1, 3.2))
ax.plot(t/wave_period, F_total, label=f'Solar Island Force (A = {wave_amplitude:.2f} m, T = {wave_period:.1f} s)')
ax.set_xlabel(r'$t/T$ [-]')
ax.set_ylabel(r'$F_{y,\mathrm{total}}$ [N/m]')

#ax.set_title('Total Wave-Induced Force on Solar Island')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, handlelength=2)
ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.5)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
ax.set_ylim(bottom=0.0)
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
ax.minorticks_on()
#ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/total_force_solar_island.pdf", transparent=True, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.2))  # slightly wider
selected_indices = [0, n_modules // 2, n_modules - 1]
colors = ['#009E73', '#0072B2', '#D55E00']
for i, idx in enumerate(selected_indices):
    ax.plot(t/wave_period, F_modules[idx, :], label=f'$i={idx+1}$, $y={x_positions[idx]:.1f}$ m', color=colors[i])

ax.set_xlabel(r'$t/T$ [-]')
ax.set_ylabel(r'$F_{y,\mathrm{module}}$ [N/m]')
ax.xaxis.labelpad = 6
ax.yaxis.labelpad = 6
ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.5)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=9)
ax.minorticks_on()
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, handlelength=1.5, fontsize=9)
plt.tight_layout(pad=1.5)
plt.savefig("plots/module_forces_individual.pdf", transparent=True, bbox_inches='tight')
plt.close()

"✅ Saved → plots/total_force_solar_island.pdf\n✅ Saved → plots/module_forces_individual.pdf"
