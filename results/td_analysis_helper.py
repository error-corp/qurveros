import filter_functions
import qutip
import numpy as np


def get_dephasing_pulse_sequence(control_dict):

    """
    Generates the pulse sequence object for dephasing noise. Code adapted from:
    https://filter-functions.readthedocs.io/en/latest/examples/periodic_driving.html
    """

    Hx = qutip.sigmax()/2
    Hy = qutip.sigmay()/2
    Hz = qutip.sigmaz()/2

    dt = np.diff(control_dict['times'])
    Tg = control_dict['times'][-1]

    omega = control_dict['omega'][1:]
    phi = control_dict['phi'][1:]
    delta = control_dict['delta'][1:]

    x_control = omega*np.cos(phi)
    y_control = omega*np.sin(phi)
    z_control = delta

    H_control = [[Hx, x_control], [Hy, y_control], [Hz, z_control]]

    H_noise = [[Hz, np.ones_like(dt)/Tg]]  # Additive dephasing noise

    dephasing_ps = filter_functions.PulseSequence(H_control, H_noise, dt)

    return dephasing_ps


def get_dephasing_ff_ps(control_dict, n_samples=500, omega_tau_max=16*np.pi):

    dephasing_ps = get_dephasing_pulse_sequence(control_dict)

    Tg = control_dict['times'][-1]

    omega = filter_functions.util.get_sample_frequencies(
                                            dephasing_ps,
                                            n_samples=n_samples,
                                            omega_max=omega_tau_max/Tg,
                                            spacing='linear')

    dephasing_ff = dephasing_ps.get_filter_function(omega, which='fidelity')

    omega_tau = omega*Tg

    dephasing_ff = (Tg**(-2))*dephasing_ff[0, 0]

    return omega_tau, dephasing_ff, dephasing_ps
