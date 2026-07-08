"""Demonstration of complete reflectometry noise simulator."""
import numpy as np
import matplotlib.pyplot as plt
#from environment import Environment
from hardware.components import Antenna, Waveguide, Receiver
from measurement import ReflectometrySystem
from scipy.fft import fft, fftfreq
from scipy import signal as sgn
#from signal_own import sinusoidal_signal
import time
from hardware.utils.utils import wavelength

# Define system components
antenna = Antenna(name="W7X Antenna", physical_temp_k=300.0, ant_pos=np.array([0.0, 0.0, 0.0]), boresight=np.array([0.0, 1.0, 0.0]), e_plane_up=np.array([0.0, 0.0, 1.0]), a1=39.97e-3, b1=30.588e-3, rho_1=30e-3, rho_2=60e-3, gain_db=19.0, radiation_efficiency=0.99, reflection_efficiency=1, PLF=1.0, resistance_ohm=50)
waveguide = Waveguide(name="WR-90 Waveguide", physical_temp_k=300.0, attenuation_db_m=0.5, length_m=20.0)
receiver = Receiver(name="Bunny Receiver", gain_db=20.0, noise_figure_db=20, RBW_hz=1e3, sampling_rate_hz=12e6, rfl_freq_hz=20e9, nquist_margin=2)
#adc = ADC("12-bit ADC", bits=12) ???
A = 0.1 # Signal amplitude (V) at antenna terminals, can be related to E-field and antenna effective area



system = ReflectometrySystem(
    blocks={"antenna": antenna, "waveguide": waveguide, "receiver": receiver},  #], adc],
    rng=42  # Reproducible
)

printing = False
if printing:
    print(f"Receiver sampling rate: {system.blocks['receiver'].sampling_rate_hz:.0e} Hz")
    print(f"Receiver duration: {system.blocks['receiver'].duration_s:.0e} s")
    print(f"Receiver number of samples: {system.blocks['receiver'].n_samples}")

    print(f"Frequency of operation: {system.blocks['receiver'].rfl_freq_hz/1e9:.0e} GHz")
    print(f"wavelength at receiver frequency: {wavelength(system.blocks['receiver'].rfl_freq_hz):.2e} m")
    print(f"Ant_a: {system.blocks['antenna'].a1*1e3:.2f} mm")
    print(f"Ant_b: {system.blocks['antenna'].b1*1e3:.2f} mm")
    print(f"Ant_rho_1: {system.blocks['antenna'].rho_1*1e3:.2f} mm")
    print(f"Ant_rho_2: {system.blocks['antenna'].rho_2*1e3:.2f} mm")

    print(f"Ant_s: {system.blocks['antenna'].phase_err_E_plane(system.blocks['receiver'].rfl_freq_hz)[0]:.2f}")
    print(f"Ant_t: {system.blocks['antenna'].phase_err_H_plane(system.blocks['receiver'].rfl_freq_hz)[0]:.2f}")

    print(f"Antenna phase_err_E_plane: {system.blocks['antenna'].phase_err_E_plane(system.blocks['receiver'].rfl_freq_hz)[1]:.2f}")
    print(f"Antenna phase_err_H_plane: {system.blocks['antenna'].phase_err_H_plane(system.blocks['receiver'].rfl_freq_hz)[1]:.2f}")
    print(f"Antenna directivity: {system.blocks['antenna'].directivity_complex_db(system.blocks['receiver'].rfl_freq_hz, 0, 0)} dB")



# Define source noise model 
environment = Environment(name="Simplified model", plasma_physical_temp_keV=10.0, PLF=0.5, resistance_ohm=2.0, source_pos=np.array([0.0, 4.33012, 2.5]))
E_field_vm = 3e-2  # V/m



# Generate measurement
noisy_trace, ideal_signal, signal, total_noise = system.measure(
    n_samples=system.blocks["receiver"].n_samples, 
    sampling_rate_hz=system.blocks["receiver"].sampling_rate_hz, 
    rfl_freq_hz=system.blocks["receiver"].rfl_freq_hz, 
    environment=environment, 
    E_field_vm=E_field_vm, 
    duration_s=system.blocks["receiver"].duration_s, 
    A=A
)

freqs_w, Pxx_w = sgn.welch(
    noisy_trace, fs=system.blocks["receiver"].sampling_rate_hz, window='hann', scaling='density'
)

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

plt.figure(figsize=(9, 4))
plt.plot(freqs_w, 10*np.log10(Pxx_w + 1e-20))
plt.title('Autospectrum (Welch)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB/Hz)')
plt.grid(True)
plt.tight_layout()
filename_ideal_signal = (
    f"Reflectometry/plots/"
    f"{timestamp}_"
    f"welch_"
    f"rfl_freq_hz{system.blocks['receiver'].rfl_freq_hz:.0e}Hz_"
    f"samplingfreq{system.blocks['receiver'].sampling_rate_hz:.0e}Hz_"
    f"RBW{system.blocks['receiver'].RBW_hz:.0e}Hz"
    f"Ant_temp{system.blocks['antenna'].physical_temp_k:.0f}K_"
    f"Ant_eff{system.blocks['antenna'].radiation_efficiency:.0f}"
    f"Ant_ref{system.blocks['antenna'].reflection_efficiency:.0f}"
    f"Ant_PLF{system.blocks['antenna'].PLF:.0f}"
    f"Ant_a1{system.blocks['antenna'].a1*1e3:.0f}mm_"
    f"Ant_b2{system.blocks['antenna'].b1*1e3:.0f}mm_"
    f"Ant_rho1{system.blocks['antenna'].rho_1*1e3:.0f}mm_"
    f"Ant_rho2{system.blocks['antenna'].rho_2*1e3:.0f}mm_"
    f"Ant_phase_err_E{system.blocks['antenna'].phase_err_E_plane(system.blocks['receiver'].rfl_freq_hz)[1]:.0f}deg_"
    f"Ant_phase_err_H{system.blocks['antenna'].phase_err_H_plane(system.blocks['receiver'].rfl_freq_hz)[1]:.0f}deg_"
    f"Ant_directivity{system.blocks['antenna'].directivity_complex_db(system.blocks['receiver'].rfl_freq_hz, 0, 0):.0f}dB"
)



print(filename_ideal_signal)
plt.savefig(filename_ideal_signal, dpi=300, bbox_inches='tight')
plt.close()

if printing:
    print(f"System noise temperature: {sum(b.input_noise_temp_k for name, b in system.blocks.items()):.1f} K")

    print("SourceNoiseTemp: {:.4f} K".format(environment.noise_temp_k))
    print("AntennaEqv_Input_NoiseTemp: {:.4f} K".format(system.blocks["antenna"].input_noise_temp_k))
    print("AntennaEqv_Output_NoiseTemp: {:.4f} K".format(system.blocks["antenna"].output_noise_temp_k))

    print("Conv: {:.4f} Np/m".format(system.blocks["waveguide"].attenuation_Np_m))
    print("WGEff: {:.4f}".format(system.blocks["waveguide"].efficiency))
    print("WGOutEqvNoiseTemp: {:.4f} K".format(system.blocks["waveguide"].output_noise_temp_k))
    print("ReceiverNoiseTemp: {:.4f} K".format(system.blocks["receiver"].input_noise_temp_k))
