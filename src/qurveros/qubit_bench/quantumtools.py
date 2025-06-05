"""
This module provides some helper functions involving quantum operations.
"""

import qutip
import numpy as np


def calculate_gate_fidelity(u_sim, u_target):

    """
    Calculates the gate fidelity between a simulated gate and a target gate.

    Args:
        u_sim (array): The simulated SU(2) gate.
        u_target (array): The target SU(2) gate.

    Returns:
        fidelity (float): The average gate fidelity between the gates.

    Note:

    There exist multiple definitions of the gate fidelity. Care must be
    taken on the factors involved in each definition.

    For convenience, we define M = u_target^{dagger} u_sim

    In our work, we used the average gate fidelity as defined in:

    Fidelity of quantum operations by Pedersen et al.
    https://doi.org/10.1016/j.physleta.2007.02.069
    """

    u_target = qutip.Qobj(u_target)
    u_sim = qutip.Qobj(u_sim)

    M = (u_target.dag())*(u_sim)

    unitary_m_check = (M * (M.dag())).tr()
    unitary_m_tr_sq = (M.tr())*(M.dag().tr())

    d = M.shape[0]
    normalization = d*(d+1)

    avg_fidelity = (unitary_m_check + unitary_m_tr_sq)/normalization

    return avg_fidelity.real


def calculate_adj_rep(gate):

    """
    Calculates the adjoint representation of a given SU(2) gate, from the
    definition of the transfer matrix.
    """

    A = 1j*np.zeros((4, 3))
    for idx, pauli in enumerate([qutip.sigmax(),
                                 qutip.sigmay(),
                                 qutip.sigmaz()]):

        A[:, idx] = qutip.operator_to_vector(pauli).full().flatten()

    gate = qutip.Qobj(gate)

    A_dag = np.conjugate(A.T)

    adj_rep = 0.5*A_dag @ (qutip.to_super(gate).full()) @ A

    return adj_rep.real
