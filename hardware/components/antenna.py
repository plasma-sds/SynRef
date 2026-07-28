"""Antenna model as noisy source."""
from typing import Tuple
from dataclasses import dataclass, field
from hardware.utils.utils import db_to_lin, wave_number, wavelength, complex_awgn, thermal_noise_rms_power, thermal_noise_rms_voltage, field_to_open_circuit_voltage
from hardware.environment import Environment
import numpy as np
from scipy import constants as const
from hardware.utils.antenna.pyramidal_horn_farfield_E_U import compute_directivity

HORN_PRESETS = {
    'default': dict(
        a1=40e-3,     # H-plane aperture width [m]
        b1=40e-3,     # E-plane aperture height [m]
        rho1=50e-3,   # E-plane slant length [m]
        rho2=50e-3,   # H-plane slant length [m]
        x=-0.1,       # horn x-position [m] (<=0)
        E1=3,
    ),
    'W7X': dict(
        a1=39.97e-3,  # H-plane aperture width [m]
        b1=30.588e-3, # E-plane aperture height [m]
        rho1=30e-3, # E-plane slant length [m]
        rho2=60e-3,  # H-plane slant length [m]
        x=-0.1,       # horn x-position [m] (<=0)
        E1=3,
    )
    # add more named presets here, e.g. 'wide', 'narrow', etc.
}

@dataclass
class Antenna:
    """Antenna modeled as source with noise only from antenna losses and physical temperature."""
    name: str
    physical_temp_k: float = 290.0  # Physical temperature of antenna
    ant_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    boresight: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    e_plane_up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    a1: float = 40e-3              # E‑plane dimension [m]
    b1: float = 30e-3                # H‑plane dimension [m]
    rho1: float = 50e-3             # E‑plane flare length [m]
    rho2: float = 40e-3             # H‑plane flare length [m]
    gain_db: float = 0.0
    radiation_efficiency: float = 0.99  # Antenna efficiency (0 to 1); losses = 1 - efficiency
    reflection_efficiency: float = 1  # Reflection efficiency (0 to 1);
    PLF: float = 1.0 # Polarization loss factor (0 to 1)
    bandwidth_hz: float = 1e9
    operating_frequency_hz: float = 30e9
    resistance_ohm: float = 50.0
    phase_error_E_coeffs: Tuple[float, ...] = (-2.73659530e+01, 2.17043653e+01, 3.48345693e+00, 1.34845221e+01, -4.83945110e-01, 1.58573717e-02)
    phase_error_H_coeffs: Tuple[float, ...] = (-4.68171544, 10.28708738, -0.31046327, 0.01313199)

    @property
    def gain_lin(self):
        return db_to_lin(self.gain_db)
    
    @property
    def loss_factor(self):
        """Loss factor L = 1 / efficiency (attenuation due to losses)."""
        return 1.0 / self.radiation_efficiency
    
    @property
    def directivity_max_lin(self):
        """Directivity D = G / efficiency."""
        return self.gain_lin / self.radiation_efficiency
    
    def phase_err_E_plane(self, rfl_freq_hz: float) -> np.ndarray:
        """Example phase error function across E-plane (x-axis). Dased on Balanis F13.23. data fitted to polynomial."""
        lam = wavelength(rfl_freq_hz)
        s = self.b1 ** 2 / (8 * lam * self.rho1)
        if s > 1:
            raise ValueError(f"s must be <= 1, got {s}")
        return s, np.polyval(np.array(self.phase_error_E_coeffs), s)

    def phase_err_H_plane(self, rfl_freq_hz: float) -> np.ndarray:
        """Example phase error function across H-plane (y-axis). Dased on Balanis F13.23. data fitted to polynomial."""
        lam = wavelength(rfl_freq_hz)
        t = self.a1 ** 2 / (8 * lam * self.rho2)
        if t > 1:
            raise ValueError(f"t must be <= 1, got {t}")
        return t, np.polyval(np.array(self.phase_error_H_coeffs), t)

    def directivity_max_complex_db(self, rfl_freq_hz: float) -> float:
        """Directivity in dB. Based on Balanis 13-53."""
        lam = wavelength(rfl_freq_hz)
        ideal_db = 10.0 * (1.008 + np.log10((self.a1 * self.b1) / lam**2))
        loss_e = self.phase_err_E_plane(rfl_freq_hz)[1]
        loss_h = self.phase_err_H_plane(rfl_freq_hz)[1]
        return float(ideal_db - (loss_e + loss_h))

    def directivity_max_complex_lin(self, rfl_freq_hz: float) -> float:
        """Directivity in linear scale."""
        return db_to_lin(self.directivity_max_complex_db(rfl_freq_hz))
    
    def directivity_complex_lin(self, rfl_freq_hz: float, theta: float, phi: float) -> float:
        """Directivity in linear scale. Based on Balanis 13-48b-c."""
        D_lin, _, _ = compute_directivity(
            theta=theta, phi=phi, a1=self.a1, b1=self.b1, rho1=self.rho1, rho2=self.rho2, freq=rfl_freq_hz, E1=1
        )
        return D_lin

    def directivity_complex_db(self, rfl_freq_hz: float, theta: float, phi: float) -> float:
        """Directivity in dB. Based on Balanis 13-48b-c."""
        _, D_db, _ = compute_directivity(
            theta=theta, phi=phi, a1=self.a1, b1=self.b1, rho1=self.rho1, rho2=self.rho2, freq=rfl_freq_hz, E1=1
        )
        return D_db

    @property
    def input_noise_temp_k(self):
        """Equivalent input noise temperature due to physical temperature and losses"""
        return self.physical_temp_k * (self.loss_factor - 1.0)
        
    @property
    def output_noise_temp_k(self):
        """Equivalent output noise temperature"""
        return self.physical_temp_k * (1.0 - self.radiation_efficiency)
    
    
    def effective_aperture_m2(self, rfl_freq_hz: float = None, theta: float = 0, phi: float = 0) -> float:
        """Ideal effective aperture area A_e = (D * lambda^2 / 4 * pi).
        Multiplied by : radiation efficiency, reflection efficiency and PLF to account for losses."""
        A_em = self.directivity_complex_lin(rfl_freq_hz, theta, phi) * wavelength(rfl_freq_hz) ** 2 / (4 * np.pi)
        return A_em * self.radiation_efficiency * self.reflection_efficiency * self.PLF

    def sophisticated_effective_area_m2(self, aperture_x_resolution: int, aperture_y_resolution: int, rfl_freq_hz: float = None) -> float:
        """Sophisticated effective area calculation.
        Args:
            aperture_x_resolution: Number of points in the x-direction
            aperture_y_resolution: Number of points in the y-direction
            rfl_freq_hz: Operating frequency (Hz)

        Returns:
            Effective aperture area (m^2) calculated from the field distribution.

        
        """
        k = wave_number(wavelength(rfl_freq_hz))
        E_amp, phase = Environment.E_field_on_aperture(self.a1, self.b1, aperture_x_resolution, aperture_y_resolution, rfl_freq_hz)
        x = np.linspace(-self.a1/2, self.a1/2, E_amp.shape[0])
        y = np.linspace(-self.b1/2, self.b1/2, E_amp.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # boresight far‑field complex amplitude
        # F0 = ∬ E(x,y) exp( j k (x sinθ cosφ + y sinθ sinφ) ) dxdy
        
        theta = np.linspace(0, np.pi / 2, 91)      # 0..90 deg
        phi = np.linspace(0, 2 * np.pi, 181)       # 0..360 deg

        F = np.zeros((len(theta), len(phi)), dtype=complex)

        for i, th in enumerate(theta):
            st = np.sin(th)
            for j, ph in enumerate(phi):
                phase_term = np.exp(1j * k * (X * st * np.cos(ph) + Y * st * np.sin(ph)))
                F[i, j] = np.trapz(np.trapz(E_amp*np.exp(1j*phase) * phase_term, dx=dy, axis=1), dx=dx, axis=0)
                P[i, j] = np.abs(F[i, j])**2

        pmax = np.max(P)
     
        P_rad = np.trapz(np.trapz(np.abs(E_amp)**2, dx=x[1]-x[0], axis=0), dx=y[1]-y[0])
        D0 = 4 * np.pi * np.abs(F[45, 90])**2 / P_rad
        wavelength_m = wavelength(rfl_freq_hz)
        return D0 * wavelength_m ** 2 / (4 * np.pi)

    def wavefront_to_voltage(self, E_field_vm: np.ndarray, rfl_freq_hz: float, 
                         n_samples: int = 1, rng=None) -> np.ndarray:
        """
        Receive 2D wavefront: E-field (V/m) -> voltage signal (V) with gain
        
        Args:
            E_field_vm: Incident electric field (V/m)
            rfl_freq_hz: Operating frequency (Hz)
            n_samples: Number of complex time samples
            rng: Random number generator
            
        Returns:
            Measured complex voltage signal (V)
        """
        # 1. Convert E-field to open-circuit voltage
        V_oc = field_to_open_circuit_voltage(E_field_vm, self.effective_aperture_m2(rfl_freq_hz), self.resistance_ohm)
        return V_oc
    
    def generate_noise(self, n_samples: int, receiver_RBW_hz: float, rng=None) -> np.ndarray:
        """Generate antenna thermal noise contribution."""
        noise_voltage = thermal_noise_rms_voltage(self.output_noise_temp_k, receiver_RBW_hz, resistance_ohm=self.resistance_ohm)
        return complex_awgn(n_samples, noise_voltage, rng)
    
    def apply_noise_and_gain(self, signal: np.ndarray, noise: np.ndarray, rng=None) -> np.ndarray:
        """Apply antenna gain and add noise to signal."""
        scaled_signal = np.sqrt(self.gain_lin) * signal
        scaled_noise = np.sqrt(self.gain_lin) * noise  # Noise is also amplified by gain
        return scaled_signal + scaled_noise
    
    def apply_gain(self, signal: np.ndarray, rfl_freq_hz: float, theta: float, phi: float) -> np.ndarray:
        """Apply antenna gain to signal."""
        return np.sqrt(self.directivity_complex_lin(rfl_freq_hz, theta, phi)) * signal
    
    def E_field_on_aperture(
        self,
        num_points_a: int,
        num_points_b: int,
        rfl_freq_hz: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the complex aperture E-field of the pyramidal horn.

        Aperture field model (Balanis 3rd ed., eq. 13-46 / Nikolova eq. 18.34):

            E_a(x', y') = E1 * cos(pi*x'/a1)
                        * exp[-jk*(sqrt(rho2^2 + x'^2) - rho2)]   <- H-plane exact spherical phase
                        * exp[-jk*(sqrt(rho1^2 + y'^2) - rho1)]   <- E-plane exact spherical phase

        Amplitude taper:
            - cos(pi*x'/a1)  in x: TE10 H-plane cosine taper  (CORRECT)
            - uniform in y          TE10 E-plane amplitude     (CORRECT, NO cosine in y)

        Phase:
            - Exact spherical wavefront, NOT the quadratic approximation x^2/(2*rho).

        Parameters
        ----------
        num_points_a : int    number of sample points along x (H-plane, width a1)
        num_points_b : int    number of sample points along y (E-plane, height b1)
        f0           : float  operating frequency [Hz]

        Returns
        -------
        X       : ndarray (num_points_a, num_points_b)  x-coordinates [m]
        Y       : ndarray (num_points_a, num_points_b)  y-coordinates [m]
        E_amp   : ndarray (num_points_a, num_points_b)  real amplitude (normalised, E1=1)
        phase   : ndarray (num_points_a, num_points_b)  phase [rad]
        """
        lam = 3e8 / rfl_freq_hz
        k   = 2.0 * np.pi / lam

        x = np.linspace(-self.a1 / 2, self.a1 / 2, num_points_a)
        y = np.linspace(-self.b1 / 2, self.b1 / 2, num_points_b)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # ------------------------------------------------------------------
        # Amplitude: TE10 cosine taper in x, UNIFORM in y  (Balanis 13-46)
        # ------------------------------------------------------------------
        E_amp = np.cos(np.pi * X / self.a1)   # purely real, ranges [-1, 1]
        
        # ------------------------------------------------------------------
        # Phase: exact spherical wavefront (no quadratic approximation)
        # rho1 = E-plane slant length (apex -> aperture centre, y-direction)
        # rho2 = H-plane slant length (apex -> aperture centre, x-direction)
        # ------------------------------------------------------------------
        phase_x = k * (np.sqrt(self.rho_2**2 + X**2) - self.rho_2)   # H-plane
        phase_y = k * (np.sqrt(self.rho_1**2 + Y**2) - self.rho_1)   # E-plane
        phase   = -(phase_x + phase_y)   # negative: wave travels away from apex

        return X, Y, E_amp, phase