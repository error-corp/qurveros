"""
This module defines tests for curves.
"""

import unittest

import jax.numpy as jnp

from qurveros.spacecurve import SpaceCurve, BezierCurve
from qurveros import beziertools, frametools

BESSEL_MAX_AMPLITUDE = 38.4203430
BEZIER_POINTS = 1.0 * jnp.array(
    [0, 0, 0, 1, 0, 0, 1, 1, 4, 2, 4, 1, 1, 0, 0, 0, 0, 0]
).reshape(-1, 3)


class PlanarTestCase(unittest.TestCase):
    """
    This test calculates the frenet_dict for a constant curvature curve.
    """

    def setUp(self):
        def tangent(x, params):
            x_comp = 0.0

            y_comp = jnp.sin(params[0] * x)
            z_comp = jnp.cos(params[0] * x)

            return jnp.array([x_comp, y_comp, z_comp])

        spacecurve = SpaceCurve(
            curve=tangent, order=1, interval=[0, jnp.pi], params=[1.0]
        )

        spacecurve.evaluate_frenet_dict()
        spacecurve.evaluate_control_dict("XY")
        self.spacecurve = spacecurve

    def test_constant_envelope(self):
        self.assertTrue(
            jnp.allclose(jnp.diff(self.spacecurve.get_control_dict()["omega"]), 0.0)
        )

    def test_envelope_amp(self):
        self.assertTrue(
            jnp.allclose(self.spacecurve.get_control_dict()["omega"][0], jnp.pi)
        )

    def test_arclength(self):
        Tg = frametools.calculate_total_length(self.spacecurve.get_frenet_dict())
        self.assertTrue(jnp.isclose(Tg, jnp.pi))

    def test_enpoint_distance(self):
        self.spacecurve.evaluate_robustness_properties()
        robustness_dict = (
            self.spacecurve.get_robustness_properties().get_robustness_dict()
        )

        r_diff = robustness_dict["closed_test"]

        self.assertTrue(jnp.isclose(r_diff[1], 2 / jnp.pi))


class BesselTestCase(unittest.TestCase):
    """
    The Bessel curve defined in the bessel_curve. This test will check the
    robustness conditions numerically along with the maximum of the envelope.
    """

    def setUp(self):
        def curve(x, a):
            theta = a[0] * jnp.cos(x)
            phi = a[1] * theta

            x_comp = jnp.cos(phi) * jnp.sin(theta)
            y_comp = jnp.sin(phi) * jnp.sin(theta)
            z_comp = jnp.cos(theta)

            return jnp.array([x_comp, y_comp, z_comp])

        spacecurve = SpaceCurve(
            curve=curve, order=1, interval=[0, 2 * jnp.pi], params=[5.5201, 0.5660]
        )

        spacecurve.evaluate_frenet_dict()
        spacecurve.evaluate_robustness_properties()
        spacecurve.evaluate_control_dict("XY")

        self.spacecurve = spacecurve
        self.robustness_dict = (
            spacecurve.get_robustness_properties().get_robustness_dict()
        )

    def test_closed(self):
        endpoint_diff = jnp.sum(self.robustness_dict["closed_test"] ** 2)
        self.assertTrue(jnp.isclose(endpoint_diff, 0.0, atol=1e-5))

    def test_curve_area_vanishes(self):
        area = jnp.sum(self.robustness_dict["curve_area_test"] ** 2)
        self.assertTrue(jnp.isclose(area, 0.0, 0.0))

    def test_tantrix_area_vanishes(self):
        tantrix_area = jnp.sum(self.robustness_dict["tantrix_area_test"] ** 2)
        self.assertTrue(jnp.isclose(tantrix_area, 0.0, 0.0))

    def test_envelope_max(self):
        Tg_kmax = jnp.max(self.spacecurve.get_control_dict()["omega"])

        self.assertTrue(jnp.isclose(Tg_kmax, BESSEL_MAX_AMPLITUDE))


class BezierTestCase(unittest.TestCase):
    """
    This test compares the Bezier curve calculations in SCQC,
    when the derivatives are found using JAX compared to the analytic
    derivatives from the properties of the Bernstein basis.
    """

    def setUp(self):
        spacecurve = SpaceCurve(
            curve=beziertools.bezier_curve_vec,
            order=0,
            interval=[0, 1],
            params=BEZIER_POINTS,
        )

        spacecurve.evaluate_frenet_dict()

        beziercurve = BezierCurve(BEZIER_POINTS)
        beziercurve.evaluate_frenet_dict()

        spacecurve.evaluate_control_dict("XY")
        beziercurve.evaluate_control_dict("XY")

        self.spacecurve = spacecurve
        self.beziercurve = beziercurve

    def test_frame_match(self):
        for i in range(3):
            with self.subTest(f"Frame index: {i}"):
                self.assertTrue(
                    jnp.allclose(
                        self.spacecurve.get_frenet_dict()["frame"][:, i, :],
                        self.beziercurve.get_frenet_dict()["frame"][:, i, :],
                    )
                )
