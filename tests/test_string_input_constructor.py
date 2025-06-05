"""
This module contains tests for using a string in the space curve constructor.
"""

import unittest
import numpy as np
import jax.numpy as jnp

from qurveros.spacecurve import SpaceCurve


class StringParsingTestCase(unittest.TestCase):
    """
    Tests the string-to-callable parsing in SpaceCurve by driving
    evaluate_frenet_dict only at the exact x-values we want, plus
    a direct call for parameter-order test to avoid Frenet NaNs.
    """

    def eval_curve(self, sc, xs):
        """
        Evaluate 'curve' via the full evaluate_frenet_dict,
        forcing it to sample exactly at xs by setting
        interval=[xs[0], xs[-1]] and n_points=len(xs).
        """
        sc.interval = [xs[0], xs[-1]]
        sc.evaluate_frenet_dict(n_points=len(xs))
        return np.array(sc.get_frenet_dict()["curve"])

    def test_python_vs_string_curve(self):
        xs = [0.1, 1.3, 2.7]
        params = [2.0, 3.0, 4.0]

        def py_curve(x, p):
            a, b, c = p
            return jnp.array([a * jnp.sin(x), b * jnp.cos(x), c * x])

        sc_py = SpaceCurve(
            curve=py_curve, order=0, interval=[xs[0], xs[-1]], params=params
        )
        sc_st = SpaceCurve(
            curve="[a*sin(x), b*cos(x), c*x]",
            order=0,
            interval=[xs[0], xs[-1]],
            params=params,
        )

        v_py = self.eval_curve(sc_py, xs)
        v_st = self.eval_curve(sc_st, xs)
        self.assertTrue(
            np.allclose(v_py, v_st),
            msg=f"Python vs. string mismatch:\n{v_py}\nvs\n{v_st}",
        )

    def test_no_param_curve(self):
        xs = [0.2, 0.7]

        sc = SpaceCurve(
            curve="[sin(x), cos(x), x]", order=0, interval=[xs[0], xs[-1]], params=[]
        )

        v = self.eval_curve(sc, xs)
        expected = np.stack([np.sin(xs), np.cos(xs), xs], axis=1)
        self.assertTrue(
            np.allclose(v, expected), msg=f"No‑param curve failed: {v} vs {expected}"
        )

    def test_extra_math_functions(self):
        xs = [0.3, 0.5]  # avoid x=0 so sqrt’s derivative is finite

        sc = SpaceCurve(
            curve="[tan(x), sqrt(x), exp(x)]",
            order=0,
            interval=[xs[0], xs[-1]],
            params=[],
        )

        v = self.eval_curve(sc, xs)
        expected = np.stack([np.tan(xs), np.sqrt(xs), np.exp(xs)], axis=1)
        self.assertTrue(
            np.allclose(v, expected), msg=f"Extra‑math curve failed: {v} vs {expected}"
        )

    def test_parameter_order_preserved(self):
        # Use a trig-based curve to ensure nonzero curvature for safe Frenet
        xs = [0.5, 0.6]
        params = [10.0, 2.0]  # b=10, a=2

        sc = SpaceCurve(
            curve="[b*sin(x), a*cos(x), a*b]",
            order=0,
            interval=[xs[0], xs[1]],
            params=params,
        )

        v = self.eval_curve(sc, xs)
        # expected: [b*sin(x), a*cos(x), a*b] at each x
        b, a = params
        expected = np.vstack([[b * np.sin(x), a * np.cos(x), a * b] for x in xs])
        self.assertTrue(
            np.allclose(v, expected), msg=f"Param order failure: {v} vs {expected}"
        )

    def test_unsafe_string_rejected(self):
        with self.assertRaises(ValueError):
            SpaceCurve(
                curve='[__import__("os").system("rm -rf /")]',
                order=0,
                interval=[0, 1],
                params=[],
            )


if __name__ == "__main__":
    unittest.main()
