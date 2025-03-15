"""
This module provides helper functions to simulate the control fields.
"""

import qutip
import numpy as np
from qurveros.qubit_bench import quantumtools
from qurveros.settings import settings


def simulate_control_dict(control_dict, u_target):

    r"""
    Simulates the control_dict that contains the fields to
    control a single-qubit Hamiltonian of the form:

    H = \frac{1}{2}(\Omega cos \Phi \sigma_x +
        \frac{1}{2}(\Omega \sin Phi \sigma_y +
        \frac{1}{2}(\Delta) \sigma_z

    Args:
        control_dict (dict): A dictionary with keys 'omega','phi','delta'.
        u_target (array): The SU(2) operator for the target quantum gate.

    Returns:
        A dictionary with entries:
            adj_final: The adjoint representation of the simulated control at
            the end of the evolution.
            adj_target: The adjoint representation of the target gate.
            avg_gate_fidelity: The average gate fidelity.
                See definition from quantumtools.calculate_gate_fidelity.
            u_final: The quantum gate implemented at the end of the evolution.
    """

    sim_dict = {}

    u_final = _single_qubit_sim(control_dict)[-1]
    avg_gate_fidelity = quantumtools.calculate_gate_fidelity(u_final, u_target)

    sim_dict['adj_final'] = quantumtools.calculate_adj_rep(u_final)
    sim_dict['adj_target'] = quantumtools.calculate_adj_rep(u_target)
    sim_dict['avg_gate_fidelity'] = avg_gate_fidelity
    sim_dict['u_final'] = u_final.full()

    return sim_dict


def _single_qubit_sim(control_dict):

    """
    Simulates the quantum evolution of a single qubit.
    See controltools.py for the specification of the control.

    The simulation is done in the unitless time: t' = t/Tg where Tg
    is the total gate time.
    """

    Tg = control_dict['times'][-1]

    Hx = qutip.sigmax()/2
    Hy = qutip.sigmay()/2
    Hz = qutip.sigmaz()/2

    omega = control_dict['omega']
    phi = control_dict['phi']
    delta = control_dict['delta']

    x_control = Tg*omega*np.cos(phi)
    y_control = Tg*omega*np.sin(phi)
    z_control = Tg*delta

    H_total = [[Hx, x_control], [Hy, y_control], [Hz, z_control]]
    H_total = qutip.QobjEvo(H_total, tlist=control_dict['times']/Tg)

    result = qutip.propagator(H_total, t=control_dict['times']/Tg, args=None,
                              options=settings.options['QUTIP_OPTIONS'])

    return result
