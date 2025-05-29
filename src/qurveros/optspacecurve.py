"""
This module extends the functionality of the SpaceCurve class by providing
optimization methods for the auxiliary parameters.
The BarqCurve class is also defined in this module.
"""

import jax
import jax.numpy as jnp
import numpy
import optax
import pandas as pd

from qurveros import beziertools, barqtools, spacecurve

from qurveros.misctools import progbar_range
from qurveros.settings import settings


class OptimizableSpaceCurve(spacecurve.SpaceCurve):

    """
    Extends the SpaceCurve class to provide optimization over the
    auxiliary parameters.

    Attributes:
        opt_loss: The optimization loss function created using the
                  prepare_optimization_loss method.
        params_history: The parameters obtained from each optimization step.
        loss_grad: The gradient of the loss function.

    Methods:
        prepare_optimization_loss: Creates the optimization loss based on
        the provided loss functions and their associated weights.
        optimize: Optimizes the curve's auxiliary parameters.
        update_params_from_opt_history: Updates the curve's parameters based
        on a chosen optimization step.

    Note:
        When an instance is indexed with a string, optimization information
        is returned. See the optimize method.
    """

    def __init__(self, *, curve, order, interval, params=None,
                 deriv_array_fun=None):

        self.opt_loss = None
        self.params_history = None
        self.loss_grad = lambda params: None

        super().__init__(curve=curve,
                         order=order,
                         interval=interval,
                         params=params,
                         deriv_array_fun=deriv_array_fun)

    def initialize_parameters(self, init_params=None):

        """
        Initializes the parameters for the optimization.
        """

        if init_params is not None:
            self.set_params(init_params)

    def prepare_optimization_loss(self, *loss_list, interval=None):

        """
        Creates the optimization loss based on a loss list.

        Args:
            loss_list(lists): The argument is a series of lists of the form
                [loss_function, weight]. The total loss is constructed as a
                linear combination of the function values multiplied by
                the respective weights.
            interval (list): The closed interval for the curve parameter where
            optimization takes place. The default value corresponds to the
            interval provided upon instantiation.
        """

        if interval is None:
            interval = self.interval

        x_values = jnp.linspace(*interval, settings.options['OPT_POINTS'])

        @jax.jit
        def opt_loss(params):

            frenet_dict = self._frenet_dict_fun(x_values, params)

            loss = 0.
            for loss_fun, weight in loss_list:
                loss = loss + weight * loss_fun(frenet_dict)

            return loss

        self.opt_loss = opt_loss
        self.loss_grad = jax.jit(jax.grad(self.opt_loss))

    def optimize(self, optimizer=None, max_iter=1000):

        """
        Optimizes the curve's auxiliary parameters using Optax.
        The parameters are updated based on the last iteration
        of the optimizer. The params_history attribute is also set.

        Args:
            optimizer (optax optimizer): The optimizer instance from Optax.
            max_iter (int): The maximum number of iterations.

        Raises: A RuntimeError exception is raised if the parameters
        are not set.

        Notes:
        (1) If an optimizer is not supplied, a simple gradient descent
            is implemented with learning_rate = 0.01.

        (2) If string-based indexing is used, optimization information
            is obtained at the optimization step which corresponds to the
            integer value of the string or the respective slice.
        """

        if optimizer is None:
            optimizer = optax.scale(-0.01)

        @jax.jit
        def step(params, opt_state):

            grads = self.loss_grad(params)

            updates, opt_state = optimizer.update(grads, opt_state, params)

            params = optax.apply_updates(params, updates)

            return params, opt_state

        params = self.params

        if params is None:
            raise RuntimeError(
                'The parameters are not set.'
                ' Use the .initialize_parameters() for initialization.')

        opt_state = optimizer.init(params)

        params_history = []

        for _ in progbar_range(max_iter, title='Optimizing parameters'):

            params_history.append(params)

            params, opt_state = step(params, opt_state)

        params_history.append(params)
        self.params_history = params_history

        self.set_params(params_history[-1])

    def get_params_history(self):
        return self.params_history

    def __getitem__(self, index):

        if isinstance(index, str):

            if ':' not in index:
                index = int(index)
                param_value = self.params_history[index]

                return {
                    'param_value': param_value,
                    'loss_value': self.opt_loss(param_value),
                    'loss_grad_value': self.loss_grad(param_value)}

            index = slice(*map(int, index.split(':')))
            param_value = jnp.array(self.params_history[index])
            return {
                'param_value': param_value,
                'loss_value': jax.vmap(self.opt_loss)(param_value),
                'loss_grad_value': jax.vmap(self.loss_grad)(param_value)}

        return super().__getitem__(index)

    def update_params_from_opt_history(self, opt_step=-1):
        self.set_params(self.params_history[opt_step])


class BarqCurve(OptimizableSpaceCurve):

    """
    The BarqCurve implements the BARQ method that provides the optimal control
    points for the Bezier curve based on a loss function.
    See the paper for the description of the Point Configuration (PC) which
    defines the PGF and the PRS. Their implementation can be found in
    barqtools.py

    Attributes:
        n_free_points: The number of free points used in BARQ.
        barq_fun: The map created with free_points, PGF and PRS to the Bezier
        control points.

    Methods:
        initialize_parameters: Initializes the various parameters involved in
        BARQ.
        get_bezier_control_points: Returns the Bezier curve control points used
        in BARQ.
        save_bezier_control_points: Stores the control points in a csv file.
        See the method implementation for details.

    Notes:

    The pgf_mod function is defined as:

    def pgf_mod(pgf_params, input_points):
        new_pgf_params = pgf_params.copy()
        ...
        return new_pgf_params

    The pgf_params provide flexibility on the gate-fixing stage of the BARQ
    method. The input points are the free points of the BARQ method.

    The prs_fun function is defined as:

    def prs_fun(prs_params, input_points):

        return internal_points

    At this step, the input points is an (6 + (n_free_points-2)) x 3 array.
    The first 6 points correspond to the control points which participate
    in the gate-fixing process and the rest are the free_points
    (the first two are excluded since they were already included in the first
    6 input points).

    A prs_fun that simply allows the free_points to pass acts as:
    prs_params, input_points -> input_points[6:, :].
    """

    def __init__(self, *, adj_target, n_free_points,
                 pgf_mod=None, prs_fun=None):

        """
        Initializes the BARQ method.

        Args:
            adj_target (array): The adjoint representation of the target
            operation.
            n_free_points (int): The number of free points
            for the BARQ method.
            pgf_mod (function): A function that modifies the PGF parameters.
            prs_fun (function): A function that enforces a particular structure
            on the internal points of the curve.
        """

        self.n_free_points = n_free_points

        barq_fun = barqtools.make_barq_fun(adj_target, pgf_mod, prs_fun)
        self.barq_fun = barq_fun

        bezier_deriv_array_fun = beziertools.make_bezier_deriv_array_fun()

        def barq_derivs_fun(x, params):

            W = barq_fun(params)

            return bezier_deriv_array_fun(x, W)

        def barq_curve(x, params):

            W = barq_fun(params)

            return beziertools.bezier_curve_vec(x, W)

        super().__init__(curve=barq_curve,
                         order=0,
                         interval=[0, 1],
                         deriv_array_fun=barq_derivs_fun)

    def _fit_curve_to_control_points(self, target_curve=None, target_points=None, 
                                   n_samples=100):
        """
        Fits free control points to a target curve using least squares method.
        
        This method constructs a Bernstein matrix and uses pseudo-inverse to find
        the control points that best approximate the target curve in the least
        squares sense. Only the internal free points are fitted, while maintaining
        the BARQ structure for gate-fixing points.
        
        Args:
            target_curve (callable, optional): A function that takes parameter t 
                in [0,1] and returns [x, y, z] coordinates.
            target_points (array, optional): Array of target points with shape 
                (n_points, 3).
            n_samples (int): Number of parameter samples to use for fitting.
                Default is 100.
        
        Returns:
            jnp.array: Array of fitted free_points with shape (n_free_points, 3).
            
        Raises:
            ValueError: If neither target_curve nor target_points is provided.
            
        Note:
            The fitting process respects the BARQ structure where the first 2
            free points are used in the gate-fixing process and only the internal
            points (indices 2 to n_free_points) are truly free for curve fitting.
        """
        
        if target_curve is None and target_points is None:
            raise ValueError("Either target_curve or target_points must be provided")
        
        # 1. Generate target points
        if target_curve is not None:
            t_vals = jnp.linspace(0, 1, n_samples)
            target_points = jnp.array([target_curve(float(t)) for t in t_vals])
        else:
            target_points = jnp.array(target_points)
            n_samples = len(target_points)
            t_vals = jnp.linspace(0, 1, n_samples)
        
        # 2. Estimate number of control points that BARQ will generate
        # Based on BARQ structure: 2 boundary + 6 gate-fixing + (n_free_points-2) internal
        n_total_control = self.n_free_points + 6
        
        # 3. Build Bernstein matrix using existing beziertools functions
        B_matrix = []
        for t in t_vals:
            row = []
            for i in range(n_total_control):
                # Use existing bernstein_poly function
                bernstein_val = beziertools.bernstein_poly(
                    jnp.array([i]), n_total_control-1, float(t)
                )[0]
                row.append(float(bernstein_val))
            B_matrix.append(row)
        
        B_matrix = jnp.array(B_matrix)  # Shape: (n_samples, n_total_control)
        
        # 4. Solve least squares system using pseudo-inverse
        control_points_fitted = jnp.linalg.pinv(B_matrix) @ target_points
        
        # 5. Extract only the relevant points for free_points structure
        # BARQ structure: [boundary_start, gate_fixing_points(6), internal_points, boundary_end]
        # We want internal points that correspond to free_points[2:]
        start_idx = 7  # After boundary_start + gate_fixing(6)
        end_idx = start_idx + (self.n_free_points - 2)  # -2 for first two gate-fixing free points
        
        if end_idx > len(control_points_fitted):
            # Fallback: use the last available points
            fitted_internal_points = control_points_fitted[-max(1, self.n_free_points-2):]
            if len(fitted_internal_points) < self.n_free_points - 2:
                # Pad with normalized random points if needed
                rng = numpy.random.default_rng(0)
                additional_points = rng.standard_normal(
                    (self.n_free_points - 2 - len(fitted_internal_points), 3)
                )
                additional_points = additional_points / numpy.linalg.norm(
                    additional_points, axis=1, keepdims=True
                )
                fitted_internal_points = jnp.vstack([
                    fitted_internal_points, 
                    jnp.array(additional_points)
                ])
        else:
            fitted_internal_points = control_points_fitted[start_idx:end_idx]
        
        # 6. Generate initial gate-fixing points (first 2 free_points)
        # These will be processed by the gate-fixing algorithm
        rng = numpy.random.default_rng(0)
        gate_fixing_points = rng.standard_normal((2, 3))
        gate_fixing_points = gate_fixing_points / numpy.linalg.norm(
            gate_fixing_points, axis=1, keepdims=True
        )
        
        # 7. Combine gate-fixing and fitted internal points
        free_points = jnp.vstack([
            jnp.array(gate_fixing_points),
            fitted_internal_points[:self.n_free_points-2]  # Ensure correct size
        ])
        
        return free_points

    def initialize_parameters(self, *, init_free_points=None,
                              init_pgf_params=None,
                              init_prs_params=None, seed=None,
                              fit_target_curve=None,
                              fit_target_points=None,
                              fit_n_samples=100):

        """
        Initializes the parameters for the BARQ method.
        The parameters are passed to the curve as a dictionary with entries
        containing the free points, the pgf parameters and the prs parameters.

        If no initial free points are provided, a total of n_free_points random
        points are drawn and normalized to unit magnitude.

        Args:
            init_free_points (array, optional): Initial free points array with 
                shape (n_free_points, 3). If provided, takes precedence over 
                curve fitting parameters.
            init_pgf_params (dict, optional): Initial PGF parameters dictionary.
                If None, default values are used.
            init_prs_params (dict, optional): Initial PRS parameters dictionary.
                If None, empty dictionary is used.
            seed (int, optional): Random seed for reproducible initialization.
                Default is None.
            fit_target_curve (callable, optional): Function that takes parameter t 
                in [0,1] and returns [x, y, z] coordinates for curve fitting.
                Cannot be used together with init_free_points.
            fit_target_points (array, optional): Array of target points with shape 
                (n_points, 3) for curve fitting. Cannot be used together with 
                init_free_points.
            fit_n_samples (int, optional): Number of parameter samples to use for 
                curve fitting. Default is 100. Only used when fitting to a curve.

        Raises:
            ValueError: If the number of free points provided upon
                instantiation do not agree with the dimensions of the initial
                points, or if conflicting initialization methods are specified.
                
        Examples:
            # Traditional random initialization
            barq_curve.initialize_parameters(seed=42)
            
            # Fit to a circular curve
            barq_curve.initialize_parameters(
                fit_target_curve=lambda t: [jnp.cos(2*jnp.pi*t), 
                                          jnp.sin(2*jnp.pi*t), 0]
            )
            
            # Fit to specific points
            points = jnp.array([[1,0,0], [0,1,0], [-1,0,0], [0,-1,0]])
            barq_curve.initialize_parameters(fit_target_points=points)
            
            # Manual initialization (original behavior)
            custom_points = jnp.array([[1,1,1], [2,2,2], ...])
            barq_curve.initialize_parameters(init_free_points=custom_points)
        """

        params = {}

        if seed is None:
            seed = 0

        rng = numpy.random.default_rng(seed)

        # Check for conflicting initialization methods
        curve_fitting_requested = (fit_target_curve is not None or 
                                 fit_target_points is not None)
        manual_points_provided = init_free_points is not None
        
        if curve_fitting_requested and manual_points_provided:
            raise ValueError(
                "Cannot use curve fitting (fit_target_curve/fit_target_points) "
                "together with manual initialization (init_free_points). "
                "Please specify only one initialization method."
            )

        # Curve fitting initialization
        if curve_fitting_requested:
            init_free_points = self._fit_curve_to_control_points(
                target_curve=fit_target_curve,
                target_points=fit_target_points,
                n_samples=fit_n_samples
            )

        # Traditional random initialization (original behavior)
        if init_free_points is None:
            init_free_points = rng.standard_normal((self.n_free_points, 3))
            init_free_points = \
                init_free_points/numpy.linalg.norm(init_free_points, axis=0)

            init_free_points = jnp.array(init_free_points)

        else:
            if init_free_points.shape[0] != self.n_free_points:
                raise ValueError('Inconsistent number of free points with'
                                 ' provided initial free points.')

        if init_pgf_params is None:

            init_pgf_params = barqtools.get_default_pgf_params_dict()

        if init_prs_params is None:
            init_prs_params = {}

        params['free_points'] = init_free_points
        params['pgf_params'] = init_pgf_params
        params['prs_params'] = init_prs_params

        return super().initialize_parameters(params)

    def evaluate_control_dict(self, n_points=None):

        """
        Evaluates the control dictionary using the TTC choice.
        """

        # If the pgf_mod fixes some parameters, they will not be automatically
        # updated upon execution. The corner case is when the barq_angle
        # is fixed, and the TTC used. That case must be handled with an
        # additional update or by masking the respective gradient update.

        super().evaluate_control_dict('TTC', n_points)

    def get_bezier_control_points(self):

        """
        Returns the control points used in BARQ.
        """

        return self.barq_fun(self.params)

    def save_bezier_control_points(self, filename):

        """
        Saves the control points of the associated Bezier curve.
        The first row contains the value of the binormal angle and the rest
        contain the control points used in BARQ.
        """

        points = self.get_bezier_control_points()
        barq_angle = self.params['pgf_params']['barq_angle']

        # We add the first line to store the binormal angle for the TTC.

        points = jnp.vstack([
            [barq_angle, -1., -1.],
            points
        ])

        df = pd.DataFrame(points)
        df.to_csv(filename, index=False, float_format='%.12f')
        