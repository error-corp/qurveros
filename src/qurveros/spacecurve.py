"""
This module contains the class definitions of Spacecurve, BezierCurve and
RobustnessProperties.
"""

import jax
import jax.numpy as jnp
import functools
import re

from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz
import numpy as np
from qurveros.settings import settings
from qurveros import controltools, frametools, beziertools, plottools
from qurveros.qubit_bench import simulator, quantumtools

class SpaceCurve:

    """
    Implements the SCQC to quantum evolution mapping.

    Each curve contains a strictly one traversal parameter (if it's unit-speed,
    that parameter corresponds to the time variable of the quantum evolution)
    and a set of auxiliary parameters that control its shape.

    Attributes:

        frenet_dict (dict): The frenet dictionary (see frametools.py).
        The entries of the frenet_dict are augmented with global curve
        elements.
        interval (list): The endpoints of the traversal parameter interval.
        params: The auxiliary parameters of the curve.
        control_dict (dict): The control dictionary (see controltools.py)
        robustness_prop (RobustnessProperties): Contains the robustness
        properties of the curve.

    Methods:

        evaluate_frenet_dict: Calculates the frenet dictionary
        at a given interval.
        evaluate_robustness_properties: Calculates the robustness properties
        of the curve.
        get_gate_fidelity: Computes the average gate fidelity based on the
        adjoint representation of SCQC.
        calculate_control_dict: Sets the control dictionary
        depending on the control mode.
        plot_position: Plots the position vector of the curve.
        plot_tantrix: Plots the normalized tangent vector of the curve.
        plot_fields: Plots the control fields depending on the plot mode.
        save_control_dict: Saves the control dictionary in a csv file.

    Note:
        When an instance is indexed, a dictionary is returned with entries:
        'curve', 'frame', 'curvature', 'torsion', 'deriv_array', 'length'.
    """

    def __init__(self, *, curve=None, order=None, interval=None, params=None,
             deriv_array_fun=None, control_dict=None, initial_rotation=None):
        """
        Initializes a spacecurve instance with the desired curve or control pulse.
        In principle, either a curve or the tangent of a curve can be used
        to describe the mapping. The 'order' argument indicates that choice.

        Args:
            curve (list or array): A three-element list arranged as
                [x,y,z] which correspond to the components of the curve.
            order (int): The order of the derivative that the curve argument
                corresponds to.
                order == 0 -> Position vector
                order == 1 -> Tangent
            interval (list): The closed interval of the curve parameter.
            params: The auxiliary parameters of the curve.
            deriv_array_fun (function): In case where the analytic derivatives
                are pre-calculated and are given in a function that returns an
                array of the form number_of_derivatives x 3, the automatic
                differentiation is sidestepped in favor of the function.
            control_dict (dict): Optional control dictionary to construct the curve
                from a pulse. If provided, curve, order, and interval are ignored.
            initial_rotation (array): Optional initial rotation matrix for frame
                alignment (defaults to identity).

        Raises:
            ValueError: If control_dict is provided with curve, order, or interval,
                        or if control_dict is not provided and curve, order, or interval are missing.
            RuntimeError: If neither curve nor deriv_array_fun is provided when control_dict is not used.
            AttributeError: If an attribute is referenced before the corresponding quantity is calculated.
        """
        self.frenet_dict = None
        self.interval = interval
        self.params = params
        self.control_dict = None
        self.robustness_props = None

        if control_dict is not None:
            if curve is not None or order is not None or interval is not None:
                raise ValueError("When control_dict is provided, curve, order, and interval should not be specified.")
            self._init_from_control_dict(control_dict, initial_rotation)
            return  # Exit after initializing from control_dict

        # Handle curve-based initialization
        if curve is None or order is None or interval is None:
            raise ValueError("When control_dict is not provided, curve, order, and interval must be specified.")

        if isinstance(curve, str):
            # Ensure characters are safe
            if re.search(r'[^0-9A-Za-z_ \[\],\+\-\*\/\(\).]', curve):
                raise ValueError("Unsafe characters in curve expression")
            
            # Get params
            raw_names = [m.group(1) for m in re.finditer(r'\b([A-Za-z_]\w*)\b', curve)]
            safe_math = {name: getattr(jnp, name) for name in dir(jnp) if not name.startswith("_")}
            reserved = set(safe_math) | {'x', 'jnp'}
            param_names = [n for n in dict.fromkeys(raw_names) if n not in reserved]

            # Build the source
            src = "def _f(x, params):\n"
            if param_names:
                src += f"    {', '.join(param_names)} = params\n"
            src += f"    return jnp.array({curve})"

            # Create the exec
            safe_globals = {"__builtins__": None, "jnp": jnp, **safe_math}
            exec(src, safe_globals)
            curve = safe_globals['_f']

        def curve_fun(x, params):
            return 1.0 * jnp.array(curve(x, params)).flatten()

        if deriv_array_fun is None:
            if curve is None:
                raise RuntimeError('Either a curve or a deriv_array function must be provided.')
            deriv_array_fun = frametools.make_deriv_array_fun(curve_fun, order)
        else:
            if curve is None and order == 0:
                raise RuntimeError('The position vector is not defined.')

        if order > 0:
            def curve_fun(x, params):
                return 0.

        @jax.jit
        @functools.partial(jax.vmap, in_axes=(0, None))
        def frenet_dict_fun(x, params):
            deriv_array = deriv_array_fun(x, params)
            frenet_dict = frametools.calculate_frenet_dict(deriv_array)
            frenet_dict['x_values'] = x
            frenet_dict['params'] = params
            frenet_dict['curve'] = curve_fun(x, params)
            return frenet_dict

        self._frenet_dict_fun = frenet_dict_fun
    def _init_from_control_dict(self, control_dict, initial_rotation=None):
        """
        Initializes the SpaceCurve from a control dictionary using the direct method.

        Args:
            control_dict (dict): Control dictionary containing 'times', 'omega', 'phi', 
                'delta', and optionally 'original_arclength', 'original_start'.
            initial_rotation (array): Optional initial rotation matrix (defaults to identity).
        """
        times = control_dict['times']
        self.interval = [times[0], times[-1]]
        self.params = None

        if initial_rotation is None:
            initial_rotation = jnp.eye(3)

        # Simulate quantum evolution to get unitary operators
        qu_evol = simulator._single_qubit_sim(control_dict)
        adj_evol = jnp.array([quantumtools.calculate_adj_rep(u) for u in qu_evol])

        # Compute Rz(-phi) for frame extraction
        phi = control_dict['phi']
        def Rz(theta):
            return jnp.array([
                [jnp.cos(theta), -jnp.sin(theta), 0],
                [jnp.sin(theta), jnp.cos(theta), 0],
                [0, 0, 1]
            ])
        Rz_minus_phi = jnp.array([Rz(-p) for p in phi])

        # Extract unadjusted frame from adjoint representation
        initial_rotation_inv = initial_rotation.T
        frame_unadjusted = jnp.einsum('tij,tjk,kl->til', Rz_minus_phi, adj_evol, initial_rotation_inv)
        T = frame_unadjusted[:, 2, :]  # Tangent vector
        N = frame_unadjusted[:, 1, :]  # Normal vector
        B = -frame_unadjusted[:, 0, :]  # Binormal vector

        # Adjust frame based on sign of omega with sign change detection
        omega = control_dict['omega']
        # Detect sign changes between consecutive points
        sign_changes = jnp.sign(omega[:-1] * omega[1:]) < 0
        f_t = jnp.ones_like(omega)
        # Propagate sign changes, starting with sign of first omega
        f_t = f_t.at[0].set(jnp.sign(omega[0]) if omega[0] != 0 else 1)
        for i in range(len(sign_changes)):
            if sign_changes[i]:
                f_t = f_t.at[i + 1:].set(-f_t[i + 1])
        N_frame = f_t[:, None] * N
        B_frame = f_t[:, None] * B
        frame = jnp.stack([T, N_frame, B_frame], axis=1)

        # Compute curvature and torsion
        kappa = jnp.abs(omega)
        delta = control_dict['delta']
        dphi_dt = jnp.gradient(phi, times)
        tau = dphi_dt - delta

        # Compute curve by integrating tangent vector (unit-speed curve) with variable time steps
        curve = cumtrapz(T, times, initial=0, axis=0)  # Shape matches times

        # Adjust curve using metadata from control_dict
        scale_factor = 1.0
        offset = jnp.zeros(3)
        if 'original_arclength' in control_dict and 'original_start' in control_dict:
            original_arclength = control_dict['original_arclength']
            original_start = control_dict['original_start']
            # Compute total arclength of reconstructed curve
            reconstructed_arclength = times[-1] - times[0]
            scale_factor = original_arclength / reconstructed_arclength if reconstructed_arclength != 0 else 1.0
            curve_scaled = curve * scale_factor
            offset = original_start - curve_scaled[0]
        curve = curve * scale_factor + offset
        # Precomputed frenet_dict for the given times
        precomputed_frenet_dict = {
            'x_values': times,
            'params': None,
            'curve': curve,
            'frame': frame,
            'curvature': kappa,
            'torsion': tau,
            'speed': jnp.ones_like(times),
            'deriv_array': None
        }

        # Define interpolation functions for flexibility in n_points
        interp_curve = interp1d(times, curve, axis=0, kind='linear', fill_value="extrapolate")
        interp_frame = interp1d(times, frame, axis=0, kind='linear', fill_value="extrapolate")
        interp_curvature = interp1d(times, kappa, kind='linear', fill_value="extrapolate")
        interp_torsion = interp1d(times, tau, kind='linear', fill_value="extrapolate")
        interp_speed = interp1d(times, jnp.ones_like(times), kind='linear', fill_value="extrapolate")

        def _frenet_dict_fun(x, params):
            x = np.array(x)  # interp1d expects NumPy arrays
            frenet_dict = {
                'x_values': x,
                'params': params,
                'curve': interp_curve(x),
                'frame': interp_frame(x),
                'curvature': interp_curvature(x),
                'torsion': interp_torsion(x),
                'speed': interp_speed(x),
                'deriv_array': None
            }
            # Convert results back to JAX arrays
            for key in frenet_dict:
                if isinstance(frenet_dict[key], np.ndarray):
                    frenet_dict[key] = jnp.array(frenet_dict[key])
            return frenet_dict

        self._frenet_dict_fun = _frenet_dict_fun

    def set_params(self, params):

        """
        Sets the default parameters of the curve.
        The associated elements are set to None so that the geometric
        quantities, the robustness properties and the control fields are
        evaluated for the new set of auxiliary parameters.
        """

        self.params = params

        self.frenet_dict = None
        self.control_dict = None
        self.robustness_props = None

    def get_params(self):
        return self.params

    def evaluate_frenet_dict(self, n_points=None):

        """
        Evaluates the frenet dictionary in a specified number of points.

        The frenet dictionary is augmented with:
            'x_values' : The values of x.
            'params' : The parameters for each value of x.
            'curve': The position vector of the curve.
            'length': The (cumulative) length of the curve.

        Args:
            n_points (int): The number of points where the frenet dictionary
            is evaluated at the given interval. The default value is drawn
            from the settings.options['CURVE_POINTS'].
        """

        if n_points is None:
            n_points = settings.options['CURVE_POINTS']

        x_values = jnp.linspace(*self.interval,  n_points)

        frenet_dict = self._frenet_dict_fun(x_values, self.params)

        if len(frenet_dict['curve'].shape) < 2:

            print('The curve will be constructed using the tangent vector.')
            curve = frametools.calculate_curve_from_tantrix(frenet_dict)
            frenet_dict['curve'] = jnp.array(curve)

        speed_int = frametools.calculate_cumulative_length(frenet_dict)
        frenet_dict['length'] = speed_int

        self.frenet_dict = frenet_dict

    def get_frenet_dict(self):
        return self.frenet_dict

    def __getitem__(self, index):

        dict_for_getitem = {}

        if self.frenet_dict is None:
            return dict_for_getitem

        for key in ['curve', 'frame', 'curvature', 'torsion',
                    'deriv_array', 'length']:
            dict_for_getitem[key] = self.frenet_dict[key][index]

        return dict_for_getitem

    def __len__(self):
        return len(self.frenet_dict['x_values'])

    def evaluate_robustness_properties(self):

        """
        Calculates and sets the robustness properties of the curve.
        See the RobustnessProperties class definition for more details.

        Raises:
             AttributeError: An exception is raised when the frenet dictionary
             is not initialized.
        """

        if self.frenet_dict is None:
            raise AttributeError('Have you initialized the frenet dict?')

        robustness_props = RobustnessProperties(self.frenet_dict)

        self.robustness_props = robustness_props

    def get_robustness_properties(self):

        if self.robustness_props is None:
            raise AttributeError(
                'Have you evaluated the robustness properties?')

        return self.robustness_props

    def evaluate_control_dict(self, control_mode, n_points=None):

        """
        Sets the control_dict attribute. The control dictionary contains
        the control fields that implement the curve to quantum evolution
        mapping.

        Args:
            control_mode (str): The control type for the Hamiltonian
            (see controltools.py for more information).

            n_points (int): The number of points where the control dictionary
            is evaluated at the given interval. The default value is drawn
            from the settings.options['SIM_POINTS'].

        Raises:
             AttributeError: An exception is raised when the frenet dictionary
             is not initialized.

        Note:
            The number of points used to calculate the curve can be different
            from the number of points used for the simulation. It is highly
            advised that the curve is sampled at a higher rate to capture all
            the fields' details.
        """

        if self.frenet_dict is None:
            raise AttributeError('Have you initialized the frenet dict?')

        if n_points is None:
            n_points = settings.options['SIM_POINTS']

        self.control_dict = controltools.calculate_control_dict(
            self.frenet_dict,
            control_mode,
            n_points=n_points)
        # Add metadata to control_dict
        if 'curve' in self.frenet_dict:
            self.control_dict['original_arclength'] = self.frenet_dict['length'][-1]
            self.control_dict['original_start'] = self.frenet_dict['curve'][0]

    def get_control_dict(self):

        return self.control_dict

    def get_gate_fidelity(self, adj_target):

        """
        Calculates the average gate fidelity defined in
        "A simple formula for the average gate fidelity of a
        quantum dynamical operation" by Michael A. Nielsen, equation (18),
        based on the adjoint representation.
        """

        return frametools.calculate_adj_fidelity(
            self.control_dict['adj_curve'],
            adj_target)

    def plot_position(self):

        """
        Plots the position vector of the curve.
        """
        if self.frenet_dict is None:
            raise AttributeError('Have you initialized the frenet dict?')

        self._plot_vector(self.frenet_dict['curve'])

    def plot_tantrix(self):

        """
        Plots the (normalized) tangent vector of the curve.
        """
        if self.frenet_dict is None:
            raise AttributeError('Have you initialized the frenet dict?')

        self._plot_vector(self.frenet_dict['frame'][:, 0, :])

    def _plot_vector(self, vector):

        Tg = self.frenet_dict['length'][-1]

        plottools.plot_curve(vector, self.frenet_dict['length']/Tg)

    def plot_fields(self, plot_mode='full'):

        """
        Plots the control fields depending on the plot mode.

        Raises:
            AttributeError: An exception is raised when a control dictionary
            is not set.
        """
        if self.control_dict is None:
            raise AttributeError('Have you chosen a control mode first?')

        plottools.plot_fields(self.control_dict, plot_mode)

    def save_control_dict(self, filename):

        """
        Saves the control fields in a csv file with entries described in
        the controltools.py module.
        """

        if self.control_dict is None:
            raise AttributeError('Have you chosen a control mode first?')

        controltools.save_control_dict(self.control_dict, filename)


class BezierCurve(SpaceCurve):

    """
    This class implements SCQC based on a Bezier curve, hence it
    requires only the control points.

    The deriv_array_fun is passed as argument to the constructor since
    the Bezier curves transfer the parameter differentiation to finite
    differences in the control points.
    """

    def __init__(self, points):

        """
        Receives the control points for the construction of the Bezier curve.

        Args:
            points (array): A 3 x M array of control points.

        Raises:
            ValueError: If the number of control points provided is not
            sufficient.
        """

        # Without the pre-calculation of the derivatives, we could use:
        # super().__init__(curve=beziertools.bezier_curve_vec,
        #                  order=0,
        #                  interval=[0, 1],
        #                  params=points)

        if points.shape[0] <= settings.options['NUM_DERIVS']:
            raise ValueError('More control points are required.'
                             ' Check NUM_DERIVS value')

        bezier_deriv_array_fun = beziertools.make_bezier_deriv_array_fun()

        # The current implementation of the bezier curve derivatives assumes
        # a sufficient number of points so that the finite differences do not
        # return an empty array.

        super().__init__(curve=beziertools.bezier_curve_vec,
                         order=0,
                         interval=[0, 1],
                         params=points,
                         deriv_array_fun=bezier_deriv_array_fun)


class RobustnessProperties:

    """
    Calculates the robustness properties associated with a given curve.

    When printed, robustness properties that appear as vectors are expressed
    as the squared norm of that vector.
    """

    def __init__(self, frenet_dict):

        self.frenet_dict = frenet_dict
        self.calculate_properties()

    def __repr__(self):

        for rob_test, value in self.robustness_dict.items():

            print_val = value**2

            if rob_test == 'CFI':
                print_val = value

            print(f"|{rob_test:^25}: \t {jnp.sum(print_val):.4e}")

        return ''

    def calculate_properties(self):

        """
        Calculates the robustness dictionary which contains all the robustness
        properties associated with the curve.
        """

        robustness_dict = {}

        Tg = self.frenet_dict['length'][-1]

        robustness_dict['closed_test'] = (self.frenet_dict['curve'][-1] -
                                          self.frenet_dict['curve'][0])/Tg

        robustness_dict['curve_area_test'] = \
            frametools.calculate_curve_area(self.frenet_dict)

        robustness_dict['tantrix_area_test'] = \
            frametools.calculate_tantrix_area(self.frenet_dict)

        robustness_dict['CFI'] = \
            frametools.calculate_cfi_value(self.frenet_dict)

        self.robustness_dict = robustness_dict

    def get_robustness_dict(self):

        return self.robustness_dict
