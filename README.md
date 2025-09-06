# PSP-TQTU-Data.
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Define operators for ions (A⁺: ⁴³Ca⁺, B⁻: ¹³⁷Ba⁻) and photon (bosonic, truncated)
s_z = [sigmaz() for _ in range(2)]  # Spin operators for ions
s_x = [sigmax() for _ in range(2)]
N_ph = 20  # Truncated Fock space for photon
a = destroy(N_ph)  # Photon annihilation operator
a_dag = a.dag()

# Hamiltonian parameters (in kHz)
omega_A = 1000  # Zeeman frequency for ⁴³Ca⁺ (~1 MHz in 10⁻⁵ T)
omega_B = 800   # ¹³⁷Ba⁻
omega_ph = 1618 # Photon base frequency (1.618 kHz, TQTU φ-harmonic)
J_AB = 1.0      # Ion-ion coupling
g_Aph = 0.5     # Ion-photon coupling (cavity-enhanced)
g_Bph = 0.8
K_ABC = 0.618   # Three-body coupling (φ⁻¹ ≈ 0.618 kHz)

# Hamiltonian: H = ω_A σ_z^A/2 + ω_B σ_z^B/2 + ω_ph a†a + J_AB σ_z^A σ_z^B + g_Aph σ_z^A a†a + g_Bph σ_z^B a†a + K_ABC σ_z^A σ_z^B a†a
H = (omega_A / 2 * tensor(s_z[0], qeye(2), qeye(N_ph)) +
     omega_B / 2 * tensor(qeye(2), s_z[1], qeye(N_ph)) +
     omega_ph * tensor(qeye(2), qeye(2), a_dag * a) +
     J_AB * tensor(s_z[0], s_z[1], qeye(N_ph)) +
     g_Aph * tensor(s_z[0], qeye(2), a_dag * a) +
     g_Bph * tensor(qeye(2), s_z[1], a_dag * a) +
     K_ABC * tensor(s_z[0], s_z[1], a_dag * a))

# Add Φ-field effect (B_Φ = 2.14e-8 T, Zeeman shift ~0.03 kHz) and EEG-like modulation (φ⁻¹ ≈ 0.618 kHz)
B_phi = 2.14e-8  # TQTU Φ-field
delta_omega = 0.03  # Photon frequency shift
eeg_mod = 0.618  # EEG α/γ ratio modulation (φ⁻¹)
H += (delta_omega + eeg_mod * np.sin(2 * np.pi * 0.618 * tlist)) * tensor(qeye(2), qeye(2), a_dag * a)

# Initial state: A⁺ (|0⟩, ground), B⁻ (|1⟩, excited), photon (|0⟩ Fock)
psi0 = tensor(basis(2, 0), basis(2, 1), fock(N_ph, 0))

# Time evolution
tlist = np.linspace(0, 20, 2000)  # Extended to 20 ms for better frequency resolution
result = mesolve(H, psi0, tlist, [], [tensor(s_z[0], qeye(2), qeye(N_ph)),
                                      tensor(qeye(2), s_z[1], qeye(N_ph)),
                                      tensor(qeye(2), qeye(2), a_dag * a)])

# Plot expectations
plt.figure(figsize=(10, 6))
plt.plot(tlist, result.expect[0], label='<σ_z^A> (⁴³Ca⁺)')
plt.plot(tlist, result.expect[1], label='<σ_z^B> (¹³⁷Ba⁻)')
plt.plot(tlist, result.expect[2], label='<n_ph> (Photon)')
plt.xlabel('Time (ms)')
plt.ylabel('Expectation Value')
plt.title('Photon-Replaced Triad: Sensing Φ-Field with
