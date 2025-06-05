"""
This module constitutes the frametools counterpart for Bezier curves and
defines some helper functions for the BARQ method.
"""

import functools
import jax
import jax.numpy as jnp
import jax.scipy.stats

from qurveros import frametools
from qurveros.settings import settings


def make_bezier_deriv_array_fun():
    """
    Creates the deriv_array function for Bezier curves.
    This method sidesteps the automatic differentiation of the Bernstein
    basis by using properties of the polynomials.

    See also frametools.make_deriv_array_fun.
    """

    NUM_DERIVS = settings.options["NUM_DERIVS"]

    @jax.jit
    def bezier_deriv_array_fun(x, W):
        deriv_list = []
        diff_coeff = W.shape[0] - 1
        W_deriv = diff_coeff * jnp.diff(W, axis=0)

        for _ in range(NUM_DERIVS - 1):
            rd_vec = bezier_curve_vec(x, W_deriv)
            deriv_list.append(rd_vec)

            diff_coeff = W_deriv.shape[0] - 1
            W_deriv = diff_coeff * jnp.diff(W_deriv, axis=0)

        rd_vec = bezier_curve_vec(x, W_deriv)
        deriv_list.append(rd_vec)

        return jnp.array(deriv_list)

    return bezier_deriv_array_fun


@jax.jit
def bezier_curve_vec(x, W):
    """
    Calculates the position vector of a curve based on the Bezier ansatz.

    Args:
        x (float): The evaluation of the curve at a point x.
        W (array): A (n_points) x 3 array of the control points.

    Returns:
        The 3D curve vector at point x.
    """

    i_vec = jnp.arange(W.shape[0])  # polynomial order n = n_points - 1

    bernstein_vec = bernstein_poly(i_vec, i_vec[-1], x).reshape(-1, 1)

    curve_vec = bernstein_vec.T @ W

    return jnp.squeeze(curve_vec, axis=0)


@jax.jit
def bezier_adj(first_point, second_point, angle):
    """
    Calculates the adjoint representation of SCQC based on two control points.

    Args:
        first_point (jax.Array): The first control point for the calculation.
        second_point (jax.Array): The second control point
        for the calculation.
        angle (float): The angle of rotation based on the value of the phase
                       field.

    Returns:
        The adjoint representation based on the control points.
    """

    tangent = first_point / frametools.vector_norm(first_point)

    binormal_bar = jnp.cross(first_point, second_point)
    binormal = binormal_bar / frametools.vector_norm(binormal_bar)

    normal = jnp.cross(binormal, tangent)

    frame = jnp.array([tangent, normal, binormal])
    R_gauge = frametools.calculate_rot_gauge(angle)

    return R_gauge @ frame


# Helper functions

# A custom jvp is required for the boundary points [0,1].


@jax.custom_jvp
@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None, None))
def bernstein_poly(i, n, x):
    return jax.scipy.stats.binom.pmf(i, n, x)


@jax.jit
def bernstein_poly_deriv(i, n, x):
    """
    Calculates the derivatives of the polynomials using the properties
    of the Bernstein basis.

    See On the Derivatives of Bernstein Polynomials:
    An Application for the Solution of High Even-Order Differential Equations
    by Doha et al.
    """

    return n * (bernstein_poly(i - 1, n - 1, x) - bernstein_poly(i, n - 1, x))


bernstein_poly.defjvps(
    # The gradients with respect to the first two arguments are not required.
    None,
    None,
    lambda xdot, primal_out, i, n, x: bernstein_poly_deriv(i, n, x) * xdot,
)
