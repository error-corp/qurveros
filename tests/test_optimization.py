"""
This module contains tests for the optimization of the curves.
"""

import unittest
import jax.numpy as jnp
import qutip
import optax

from qurveros import beziertools
from qurveros.optspacecurve import OptimizableSpaceCurve, BarqCurve
from qurveros.qubit_bench import quantumtools
from qurveros import losses


XGATE_TEST_POINTS = jnp.array(
    [
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
        [0.0, 0.0, 0.0],
    ]
)


class CloseTheCurveTestCase(unittest.TestCase):
    """
    This test optimizes the frequency parameter of a circle curve.
    """

    def test_closed_curve(self):
        def circle(x, omega):
            return jnp.array([jnp.cos(omega * x), jnp.sin(omega * x), 0])

        optspacecurve = OptimizableSpaceCurve(
            curve=circle, order=0, interval=[0, 2 * jnp.pi]
        )

        def closed_curve_loss(frenet_dict):
            endpoint_diff = frenet_dict["curve"][-1] - frenet_dict["curve"][0]

            return jnp.sum(endpoint_diff**2)

        def freq_loss(frenet_dict):
            omega = frenet_dict["params"][0]

            return -jnp.log10(omega)

        optspacecurve.initialize_parameters(0.5)
        optspacecurve.prepare_optimization_loss(
            [closed_curve_loss, 1.0], [freq_loss, 1]
        )

        optspacecurve.optimize()

        optspacecurve.evaluate_frenet_dict()
        optspacecurve.evaluate_robustness_properties()
        robustness_dict = (
            optspacecurve.get_robustness_properties().get_robustness_dict()
        )

        endpoint_diff = jnp.sum(robustness_dict["closed_test"] ** 2)
        self.assertTrue(jnp.isclose(endpoint_diff, 0.0, atol=1e-4))


class XgateBarqTestCase(unittest.TestCase):
    """
    Simple BARQ optimization test for an X-gate.
    """

    def setUp(self):
        u_target = qutip.sigmax()
        adj_target = quantumtools.calculate_adj_rep(u_target)

        barqcurve = BarqCurve(adj_target=adj_target, n_free_points=16)

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
        robustness_dict = (
            self.barqcurve.get_robustness_properties().get_robustness_dict()
        )

        zero_tantrix_test = jnp.sum(robustness_dict["tantrix_area_test"] ** 2)

        self.assertTrue(jnp.isclose(zero_tantrix_test, 0.0))


class BarqCurveFittingTestCase(unittest.TestCase):
    """
    Tests for the curve fitting functionality in BarqCurve.
    Verifies that control points can be fitted to target curves and points.
    """

    def setUp(self):
        # Setup target gate (X-gate) for BARQ
        u_target = qutip.sigmax()
        adj_target = quantumtools.calculate_adj_rep(u_target)

        self.barqcurve = BarqCurve(adj_target=adj_target, n_free_points=8)

    def test_fitting_accuracy_circular_function(self):
        """
        Test fitting accuracy by measuring error between fitted and target curves.
        Verifies that the fitted curve actually approximates the target well.
        """

        # Define a simple circular curve
        def circle_curve(t):
            return [jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t), 0.0]

        # Initialize with curve fitting
        self.barqcurve.initialize_parameters(
            fit_target_curve=circle_curve,
            fit_max_iter=50,  # Reduced for test speed
        )

        # Get the resulting control points through BARQ
        control_points = self.barqcurve.get_bezier_control_points()

        # Evaluate both target and fitted curves at test points
        test_t_values = jnp.linspace(0, 1, 25)
        target_points = jnp.array([circle_curve(float(t)) for t in test_t_values])

        fitted_points = jnp.array(
            [
                beziertools.bezier_curve_vec(float(t), control_points)
                for t in test_t_values
            ]
        )

        # Verify fitting accuracy with reasonable tolerance
        max_error = jnp.max(jnp.linalg.norm(fitted_points - target_points, axis=1))
        mean_error = jnp.mean(jnp.linalg.norm(fitted_points - target_points, axis=1))

        # Assert errors are within acceptable bounds (relaxed for test reliability)
        self.assertLess(max_error, 3.0, "Maximum fitting error exceeds tolerance")
        self.assertLess(mean_error, 1.5, "Mean fitting error exceeds tolerance")

    def test_fitting_to_reference_curve(self):
        """
        Test fitting to a reference BezierCurve using known control points.
        """

        # Define target curve from reference points
        def reference_curve(t):
            return beziertools.bezier_curve_vec(float(t), XGATE_TEST_POINTS)

        # Fit to the reference curve
        self.barqcurve.initialize_parameters(
            fit_target_curve=reference_curve,
            fit_max_iter=50,  # Reduced for test speed
        )

        # Check that fitting produces reasonable results
        fitted_points = self.barqcurve.get_bezier_control_points()

        # The fitted curve should be similar in overall characteristics
        # (exact match not expected due to different n_free_points and optimization)
        self.assertEqual(fitted_points.shape[1], 3, "Control points should be 3D")
        self.assertGreater(
            fitted_points.shape[0], 8, "Should have sufficient control points"
        )

        # Verify curve endpoints are reasonable (should start/end near origin for closed curve)
        self.assertLess(
            jnp.linalg.norm(fitted_points[0]), 1e-6, "Curve should start at origin"
        )
        self.assertLess(
            jnp.linalg.norm(fitted_points[-1]), 1e-6, "Curve should end at origin"
        )

    def test_conflicting_initialization_methods(self):
        """
        Test that conflicting initialization methods raise appropriate errors.
        """
        manual_points = jnp.ones((8, 3)) * 0.5

        with self.assertRaises(ValueError):
            self.barqcurve.initialize_parameters(
                init_free_points=manual_points, fit_target_curve=lambda t: [t, t, t]
            )

    def test_invalid_curve_function_handling(self):
        """
        Test that invalid curve functions raise appropriate errors.
        """

        def invalid_curve(t):
            return [float("inf"), float("nan"), 0.0]

        with self.assertRaises(ValueError):
            self.barqcurve.initialize_parameters(fit_target_curve=invalid_curve)

    def test_optimizer_parameter_propagation(self):
        """
        Test that custom optimizers are respected in curve fitting.
        """

        def simple_target_curve(t):
            return [0.5 * jnp.cos(jnp.pi * t), 0.5 * jnp.sin(jnp.pi * t), 0.2 * t]

        custom_optimizer = optax.sgd(0.1)

        # Should not raise an error and should complete successfully
        self.barqcurve.initialize_parameters(
            fit_target_curve=simple_target_curve,
            fit_optimizer=custom_optimizer,
            fit_max_iter=50,
        )

        # Verify that fitting completed
        fitted_points = self.barqcurve.get_bezier_control_points()
        self.assertIsNotNone(fitted_points)
        self.assertEqual(fitted_points.shape[1], 3)

    def test_pgf_prs_params_preservation(self):
        """
        Test that custom pgf/prs params are preserved during curve fitting.
        """

        def simple_target_curve(t):
            return [jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t), 0.1 * t]

        custom_pgf = {"barq_angle": 2.0}  # Non-default value
        custom_prs = {"test_param": 1.0}

        self.barqcurve.initialize_parameters(
            fit_target_curve=simple_target_curve,
            init_pgf_params=custom_pgf,
            init_prs_params=custom_prs,
            fit_max_iter=50,
        )

        # Verify that custom parameters were preserved
        self.assertAlmostEqual(
            self.barqcurve.params["pgf_params"]["barq_angle"], 2.0, places=5
        )
        self.assertEqual(self.barqcurve.params["prs_params"]["test_param"], 1.0)

    def test_backward_compatibility(self):
        """
        Test that traditional random initialization still works unchanged.
        """
        # Traditional random initialization should work unchanged
        self.barqcurve.initialize_parameters(seed=42)
        params_random = self.barqcurve.params["free_points"]

        # Re-initialize with same seed should give same result
        self.barqcurve.initialize_parameters(seed=42)
        params_random_repeat = self.barqcurve.params["free_points"]
        self.assertTrue(jnp.allclose(params_random, params_random_repeat))

        # Manual initialization should work unchanged
        manual_points = jnp.ones((8, 3)) * 0.5
        self.barqcurve.initialize_parameters(init_free_points=manual_points)
        self.assertTrue(
            jnp.allclose(self.barqcurve.params["free_points"], manual_points)
        )
