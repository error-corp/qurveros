"""
This module tests the noise experiments in the qubit_bench module.
"""

from qurveros.qubit_bench import noise_experiments, quantumtools
from qurveros import controltools
from qurveros.settings import settings
import unittest
import numpy as np
import qutip


class FidelityTestCase(unittest.TestCase):

    def test_unit_fidelity(self):
        fid = quantumtools.calculate_gate_fidelity(qutip.sigmax(),
                                                   [[0, 1], [1, 0]])

        self.assertTrue(np.isclose(fid, 1))

    def test_min_fidelity(self):

        fid = quantumtools.calculate_gate_fidelity(qutip.sigmax(),
                                                   qutip.sigmay())

        d = 2
        self.assertTrue(np.isclose(fid, 1/(d+1)))


class RobustnessTestCase(unittest.TestCase):

    def setUp(self):

        control_dict = controltools.make_dummy_dict(
            settings.options['SIM_POINTS'])

        control_dict['omega'] = np.pi*np.ones_like(control_dict['omega'])

        self.u_target = qutip.sigmax()
        self.control_dict = control_dict

    def test_static_additive_dephasing(self):

        sim_dict = noise_experiments.static_dephasing_experiment(
            self.control_dict,
            self.u_target)

        tg_rabi_rate = self.control_dict['omega'][0]
        tg_delta_z_vec = sim_dict['tg_delta_z']

        analytic_infid = calculate_analytic_dephasing_infid(
                                                      tg_rabi_rate,
                                                      tg_delta_z_vec)

        infid_diff = analytic_infid - sim_dict['infidelity']

        self.assertTrue(np.allclose(infid_diff, 0.))

    def test_simult_source_noise(self):

        sim_dict = noise_experiments.simult_source_experiment(
            self.control_dict,
            self.u_target)

        analytic_infid_mat = np.zeros_like(sim_dict['infidelity_matrix'])
        pulse_error_vec = sim_dict['epsilon']
        tg_delta_z_vec = sim_dict['tg_delta_z']

        for pulse_idx, pulse_error in enumerate(pulse_error_vec):
            tg_rabi_rate = (1+pulse_error)*self.control_dict['omega'][0]

            analytic_infid_mat[:, pulse_idx] = \
                calculate_analytic_dephasing_infid(tg_rabi_rate,
                                                   tg_delta_z_vec)

        infid_diff = analytic_infid_mat - sim_dict['infidelity_matrix']

        self.assertTrue(np.allclose(infid_diff, 0.))


def calculate_analytic_dephasing_infid(tg_rabi_rate, tg_delta_z_vec):

    d = 2

    theta = np.sqrt(tg_rabi_rate**2 + tg_delta_z_vec**2)
    nx_comp = tg_rabi_rate/theta

    trM_sq = (d**2)*(nx_comp*np.sin(theta/2))**2
    analytic_fid = 1/(d*(d+1))*(d+trM_sq)
    analytic_infid = 1-analytic_fid

    return analytic_infid
