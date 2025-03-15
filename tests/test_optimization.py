"""
This module contains tests for the optimization of the curves.
"""

import unittest

from qurveros.optspacecurve import OptimizableSpaceCurve, BarqCurve
from qurveros.qubit_bench import quantumtools
from qurveros import losses

import qutip
import jax.numpy as jnp


XGATE_TEST_POINTS = jnp.array([
    [0.0, 0.0, 0.0],
    [-0.9174368396882462, 0.20027364852868498, 0.21369406409562192],
    [-0.8041761519991039, 0.17554918775154812, 0.1873128075256522],
    [1.0143787013738685, 0.17178532468454982, 0.2736775539730863],
    [0.38083479899988426, 0.5492027916657358, -0.0006194580232775966],
    [-0.5354359751936735, 0.2523799047994231, 0.17442024487353],
    [-0.9418994108358996, 0.3539711361917974, -0.232999217817437],
    [-0.5711036813837912, -0.05713223886502761, 0.0018145161138630609],
    [-0.1905484142468627, 0.14036688528159752, -0.04564339757313231],
    [0.21930072573262832, -0.688227292792582, -0.004032082528444142],
    [0.3428011852341114, -0.4263672394739059, -0.4215955544900982],
    [0.13930670327019212, -0.43162490061771347, -0.18005032077844998],
    [0.2537454430731247, -0.07892925578860352, -0.3437089732582968],
    [0.6188165455173736, 0.2788967540319768, -0.1954681209080432],
    [0.23319152157880507, 0.22028400307332546, 0.007186201125869842],
    [0.5468775591637357, -0.27531072025575787, 0.40647697079809975],
    [0.31340505721501555, 0.24480221323268164, 0.24650669307439751],
    [-0.24996005966023707, 0.313430826389596, 0.9184770058348332],
    [-0.43575661910587743, -0.7757470303594219, -0.417375597747747],
    [-0.9816177358001005, 0.21428414132133258, 0.2286434054934467],
    [-0.9016288172030048, 0.19682280569985486, 0.21001197893831114],
    [0.0, 0.0, 0.0]])


class CloseTheCurveTestCase(unittest.TestCase):

    """
    This test optimizes the frequency parameter of a circle curve.
    """

    def test_closed_curve(self):

        def circle(x, omega):
            return jnp.array([jnp.cos(omega*x), jnp.sin(omega*x), 0])

        optspacecurve = OptimizableSpaceCurve(curve=circle,
                                              order=0,
                                              interval=[0, 2*jnp.pi])

        def closed_curve_loss(frenet_dict):

            endpoint_diff = frenet_dict['curve'][-1] - frenet_dict['curve'][0]

            return jnp.sum(endpoint_diff**2)

        def freq_loss(frenet_dict):

            omega = frenet_dict['params'][0]

            return -jnp.log10(omega)

        optspacecurve.initialize_parameters(0.5)
        optspacecurve.prepare_optimization_loss(
            [closed_curve_loss, 1.],
            [freq_loss, 1]
        )

        optspacecurve.optimize()

        optspacecurve.evaluate_frenet_dict()
        optspacecurve.evaluate_robustness_properties()
        robustness_dict = optspacecurve.get_robustness_properties().\
            get_robustness_dict()

        endpoint_diff = jnp.sum(robustness_dict['closed_test']**2)
        self.assertTrue(jnp.isclose(endpoint_diff, 0., atol=1e-4))


class XgateBarqTestCase(unittest.TestCase):

    """
    Simple BARQ optimization test for an X-gate.
    """

    def setUp(self):

        u_target = qutip.sigmax()
        adj_target = quantumtools.calculate_adj_rep(u_target)

        barqcurve = BarqCurve(adj_target=adj_target,
                              n_free_points=16)

        barqcurve.prepare_optimization_loss([losses.tantrix_zero_area_loss, 1])
        barqcurve.initialize_parameters(seed=0)
        barqcurve.optimize(max_iter=100)

        barqcurve.evaluate_frenet_dict()
        barqcurve.evaluate_robustness_properties()
        self.barqcurve = barqcurve

    def test_final_points_match(self):

        Xgate_points = self.barqcurve.get_bezier_control_points()
        self.assertTrue(jnp.allclose(Xgate_points, XGATE_TEST_POINTS))

    def test_tantrix_area(self):

        robustness_dict = self.barqcurve.get_robustness_properties().\
            get_robustness_dict()

        zero_tantrix_test = jnp.sum(robustness_dict['tantrix_area_test']**2)

        self.assertTrue(jnp.isclose(zero_tantrix_test, 0.))
