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
                                n_samples=100, track_losses=False):
        """
        Fits free control points to a target curve using BARQ structure.
        
        This method optimizes free_points that are then mapped through barq_fun
        to generate proper BARQ control points. It respects the gate-fixing 
        structure and PGF/PRS transformations.
        
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
        """
        
        if target_curve is None and target_points is None:
            raise ValueError("Either target_curve or target_points must be provided")
        
        # 1. Generate target points (outside JIT to avoid tracer issues)
        t_vals = jnp.linspace(0, 1, n_samples)
        
        if target_curve is not None:
            # Evaluate target curve at concrete values (outside JIT)
            target_points = []
            for i in range(n_samples):
                t_val = float(t_vals[i])
                try:
                    point = target_curve(t_val)
                    # Handle potential nan/inf values
                    point = jnp.array(point)
                    if jnp.any(jnp.isnan(point)) or jnp.any(jnp.isinf(point)):
                        # Replace with zeros if invalid
                        point = jnp.zeros(3)
                    target_points.append(point)
                except:
                    # Fallback for any evaluation errors
                    target_points.append(jnp.zeros(3))
            target_points = jnp.array(target_points)
        else:
            target_points = jnp.array(target_points)
            # If we have discrete points, interpolate to get the desired number of samples
            if len(target_points) != n_samples:
                # Simple linear interpolation for parameter values
                original_t = jnp.linspace(0, 1, len(target_points))
                target_points_interp = []
                for i in range(3):  # x, y, z components
                    target_points_interp.append(
                        jnp.interp(t_vals, original_t, target_points[:, i])
                    )
                target_points = jnp.array(target_points_interp).T
        
        # 2. Subtract initial point to fit displacement curve (as suggested in feedback)
        target_displacement = target_points - target_points[0]
        
        # 3. Initialize default PGF and PRS parameters (outside JIT)
        default_pgf_params = barqtools.get_default_pgf_params_dict()
        default_prs_params = {}
        
        # 4. Create JAX-compatible curve evaluation function
        def evaluate_curve_at_params(free_points):
            """Evaluate BARQ curve at t_vals using vectorized operations."""
            # Construct full parameter dictionary
            params = {
                'free_points': free_points,
                'pgf_params': default_pgf_params,
                'prs_params': default_prs_params
            }
            
            # Get control points through BARQ mapping
            control_points = self.barq_fun(params)
            
            # Vectorized evaluation using vmap
            def eval_single_t(t):
                return beziertools.bezier_curve_vec(t, control_points)
            
            # Use vmap for vectorized evaluation
            curve_points = jax.vmap(eval_single_t)(t_vals)
            
            return curve_points
        
        # 5. Define the loss function (no @jax.jit decorator here)
        def fitting_loss(free_points):
            # Evaluate curve
            curve_points = evaluate_curve_at_params(free_points)
            
            # Compute displacement from initial point
            curve_displacement = curve_points - curve_points[0]
            
            # Compute mean squared error between displacements
            mse = jnp.mean(jnp.sum((curve_displacement - target_displacement)**2, axis=1))
            
            return mse
        
        # 6. Initialize free_points with reasonable random values
        rng = numpy.random.default_rng(0)
        initial_free_points = rng.standard_normal((self.n_free_points, 3))
        initial_free_points = initial_free_points / numpy.linalg.norm(
            initial_free_points, axis=1, keepdims=True
        )
        initial_free_points = jnp.array(initial_free_points)
        
        # 7. Optimize using simple gradient descent (more stable than optax for this case)
        learning_rate = 0.01
        n_iterations = 500
        
        # JIT compile the gradient function separately
        grad_fn = jax.jit(jax.grad(fitting_loss))
        
        free_points = initial_free_points
        loss_history = []  # ← Agregar esta línea
        
        for i in range(n_iterations):
            try:
                grads = grad_fn(free_points)
                # Simple gradient descent update
                free_points = free_points - learning_rate * grads
                
                # Track losses if requested
                if track_losses and i % 10 == 0:  # ← Agregar estas líneas
                    current_loss = fitting_loss(free_points)
                    loss_history.append(float(current_loss))
                
            except:
                break
        
        # Return with optional loss history
        if track_losses:
            return free_points, loss_history
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
        