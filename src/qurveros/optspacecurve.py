
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
import warnings
import logging

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

    def _is_params_valid(self, params, validation_keys=None):
        """
        Checks if parameters contain NaN or infinite values.
        
        Args:
            params: Parameter tree to validate
            validation_keys: List of keys to validate. If None, validates all parameters.
            
        Returns:
            bool: True if specified parameters are finite, False otherwise
        """
        def is_finite_leaf(x):
            return jnp.all(jnp.isfinite(x))
        
        if validation_keys is None:
            # Validate all parameters
            finite_flags = jax.tree_util.tree_map(is_finite_leaf, params)
            return jax.tree_util.tree_reduce(jnp.logical_and, finite_flags, True)
        else:
            # Validate only specified keys
            if isinstance(params, dict):
                for key in validation_keys:
                    if key in params:
                        param_subset = params[key]
                        finite_flags = jax.tree_util.tree_map(is_finite_leaf, param_subset)
                        is_valid = jax.tree_util.tree_reduce(jnp.logical_and, finite_flags, True)
                        if not is_valid:
                            return False
                return True
            else:
                # For non-dict params, validate all if keys are specified
                finite_flags = jax.tree_util.tree_map(is_finite_leaf, params)
                return jax.tree_util.tree_reduce(jnp.logical_and, finite_flags, True)

    def _create_perturbed_params(self, base_params, magnitude, rng_key, validation_keys=None):
        """
        Creates perturbed parameters from valid base parameters.
        
        Args:
            base_params: Valid parameter tree to perturb
            magnitude: Magnitude of perturbation
            rng_key: JAX random key
            validation_keys: List of keys to perturb. If None, perturbs all parameters.
            
        Returns:
            Perturbed parameter tree
        """
        def add_noise_to_leaf(leaf, key):
            noise = jax.random.normal(key, leaf.shape) * magnitude
            return leaf + noise
        
        if validation_keys is None:
            # Perturb all parameters
            treedef = jax.tree_util.tree_structure(base_params)
            num_leaves = treedef.num_leaves
            keys = jax.random.split(rng_key, num_leaves)
            
            perturbed_params = jax.tree_util.tree_map(
                add_noise_to_leaf, base_params, 
                jax.tree_util.tree_unflatten(treedef, keys)
            )
            return perturbed_params
        else:
            # Perturb only specified keys
            if isinstance(base_params, dict):
                perturbed_params = dict(base_params)  # Copy base params
                
                for i, key in enumerate(validation_keys):
                    if key in base_params:
                        key_rng = jax.random.fold_in(rng_key, i)
                        param_subset = base_params[key]
                        
                        treedef = jax.tree_util.tree_structure(param_subset)
                        num_leaves = treedef.num_leaves
                        keys = jax.random.split(key_rng, num_leaves)
                        
                        perturbed_subset = jax.tree_util.tree_map(
                            add_noise_to_leaf, param_subset,
                            jax.tree_util.tree_unflatten(treedef, keys)
                        )
                        perturbed_params[key] = perturbed_subset
                        
                return perturbed_params
            else:
                # For non-dict params, perturb all if keys are specified
                return self._create_perturbed_params(base_params, magnitude, rng_key, None)

    def _log_retry_attempt(self, iteration, retry_count, error_type):
        """
        Logs retry attempts based on verbosity settings.
        
        Args:
            iteration: Current optimization iteration
            retry_count: Number of retry attempt
            error_type: Type of numerical error detected
        """
        verbosity = settings.options.get('OPTIMIZATION_VERBOSITY', 0)
        
        if verbosity >= 1:
            message = (f"Numerical instability detected at iteration {iteration}. "
                      f"Retry attempt {retry_count}: {error_type}")
            warnings.warn(message, RuntimeWarning, stacklevel=4)
            
        if verbosity >= 2:
            logger = logging.getLogger(__name__)
            logger.debug(f"Optimization retry: iter={iteration}, "
                        f"attempt={retry_count}, error={error_type}")

    def optimize(self, optimizer=None, max_iter=1000, param_validation_keys=None):

        """
        Optimizes the curve's auxiliary parameters using Optax.
        The parameters are updated based on the last iteration
        of the optimizer. The params_history attribute is also set.

        This method includes automatic handling of numerical instabilities
        that may occur during optimization. When NaN or infinite values
        are detected in the parameters, the optimizer will attempt to
        recover by perturbing the last known valid parameters and retrying.

        Args:
            optimizer (optax optimizer): The optimizer instance from Optax.
            max_iter (int): The maximum number of iterations.
            param_validation_keys (list): List of parameter keys to validate and perturb.
                If None, validates all parameters. For BarqCurve, typically ['free_points'].

        Raises: 
            RuntimeError: If the parameters are not set.

        Notes:
        (1) If an optimizer is not supplied, a simple gradient descent
            is implemented with learning_rate = 0.01.

        (2) If string-based indexing is used, optimization information
            is obtained at the optimization step which corresponds to the
            integer value of the string or the respective slice.
            
        (3) Numerical stability is handled automatically using settings:
            - MAX_OPTIMIZATION_RETRIES: Maximum retry attempts (default: 3)
            - PERTURBATION_MAGNITUDE: Size of parameter perturbation (default: 1e-6)
            - OPTIMIZATION_VERBOSITY: Logging level (default: 0)
            - NUMERICAL_CHECK_FREQUENCY: Check frequency (default: 1)
        """

        # Load numerical stability settings
        max_retries = settings.options.get('MAX_OPTIMIZATION_RETRIES', 3)
        perturbation_mag = settings.options.get('PERTURBATION_MAGNITUDE', 1e-6)
        verbosity = settings.options.get('OPTIMIZATION_VERBOSITY', 0)
        check_freq = settings.options.get('NUMERICAL_CHECK_FREQUENCY', 1)
        
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

        # Initialize numerical stability tracking
        last_valid_params = params
        last_valid_opt_state = optimizer.init(params)
        rng_key = jax.random.PRNGKey(0)
        
        # Check initial parameters
        if not self._is_params_valid(params, param_validation_keys):
            raise RuntimeError(
                'Initial parameters contain NaN or infinite values. '
                'Please check parameter initialization.')

        opt_state = optimizer.init(params)
        params_history = []

        for iteration in progbar_range(max_iter, title='Optimizing parameters'):

            # Periodic numerical stability check
            if iteration % check_freq == 0 and not self._is_params_valid(params, param_validation_keys):
                
                recovery_successful = False
                
                for retry in range(max_retries):
                    self._log_retry_attempt(iteration, retry + 1, "NaN/Inf detected")
                    
                    # Perturb last valid parameters
                    rng_key, subkey = jax.random.split(rng_key)
                    params = self._create_perturbed_params(
                        last_valid_params, perturbation_mag, subkey, param_validation_keys)
                    opt_state = last_valid_opt_state
                    
                    # Check if perturbation resolved the issue
                    if self._is_params_valid(params, param_validation_keys):
                        recovery_successful = True
                        if verbosity >= 1:
                            warnings.warn(
                                f"Successfully recovered from numerical instability "
                                f"at iteration {iteration} after {retry + 1} attempts.",
                                RuntimeWarning, stacklevel=2)
                        break
                
                if not recovery_successful:
                    # All retry attempts failed
                    error_message = (
                        f"Critical numerical instability at iteration {iteration}. "
                        f"Failed to recover after {max_retries} attempts. "
                        f"Optimization results may be unreliable. "
                        f"Consider: (1) checking initial parameters, "
                        f"(2) reducing learning rate, (3) using different optimizer, "
                        f"(4) increasing PERTURBATION_MAGNITUDE in settings."
                    )
                    
                    if verbosity >= 0:  # Always warn for critical failures
                        warnings.warn(error_message, RuntimeWarning, stacklevel=2)
                    
                    # Continue with last valid parameters as fallback
                    params = last_valid_params
                    opt_state = last_valid_opt_state

            params_history.append(params)

            # Perform optimization step
            try:
                new_params, new_opt_state = step(params, opt_state)
                last_valid_params = params
                last_valid_opt_state = opt_state
                params = new_params
                opt_state = new_opt_state
                    
            except Exception as e:
                # Optimization step failed completely
                if verbosity >= 1:
                    warnings.warn(
                        f"Optimization step failed at iteration {iteration}: {e}. "
                        f"Using last valid parameters.",
                        RuntimeWarning, stacklevel=2)
                # Keep current valid parameters
                params = last_valid_params
                opt_state = last_valid_opt_state

        if self._is_params_valid(params, param_validation_keys):
            params_history.append(params)            
        else:
            params_history.append(last_valid_params)
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

    def optimize(self, optimizer=None, max_iter=1000):
        """
        Optimizes the BARQ curve's parameters with selective validation.
        
        Only the free_points are validated and perturbed during numerical
        stability recovery, preserving the gate-fixing constraints.
        
        Args:
            optimizer: Optax optimizer instance
            max_iter: Maximum number of iterations
        """
        return super().optimize(
            optimizer=optimizer, 
            max_iter=max_iter, 
            param_validation_keys=['free_points']
        )

    def _validate_curve_point(self, point, t_val):
        """
        Validates that a curve point is valid (no NaN/Inf values).
        
        Args:
            point (array): The curve point to validate
            t_val (float): The parameter value where point was evaluated
            
        Returns:
            array: The validated point
            
        Raises:
            ValueError: If point contains NaN or Inf values
        """
        # Validate BEFORE converting to JAX array to avoid JAX errors
        point_numpy = numpy.array(point)
        if numpy.any(numpy.isnan(point_numpy)) or numpy.any(numpy.isinf(point_numpy)):
            raise ValueError(f"Target curve returned invalid values (NaN/Inf) at t={t_val}. "
                           f"Point: {point}. Check your target_curve function.")
        return jnp.array(point)

    def _generate_target_points_for_optimization(self, target_curve, target_points, n_opt_points):
        """
        Generates target points compatible with the optimization infrastructure.
        Uses the same number of points as the optimization infrastructure (OPT_POINTS).
        
        Args:
            target_curve (callable, optional): Function t -> [x,y,z] for t in [0,1]
            target_points (array, optional): Array of target points
            n_opt_points (int): Number of optimization points (from settings.options['OPT_POINTS'])
            
        Returns:
            jnp.array: Target points with shape (n_opt_points, 3)
        """
        t_vals = jnp.linspace(0, 1, n_opt_points)
        
        if target_curve is not None:
            # Evaluate target curve at optimization points
            target_points_list = []
            for i in range(n_opt_points):
                t_val = float(t_vals[i])
                point = target_curve(t_val)
                point = self._validate_curve_point(point, t_val)
                target_points_list.append(point)
            target_points = jnp.array(target_points_list)
        else:
            target_points = jnp.array(target_points)
            # Interpolate discrete points to match optimization points
            if len(target_points) != n_opt_points:
                original_t = jnp.linspace(0, 1, len(target_points))
                target_points_interp = []
                for i in range(3):  # x, y, z components
                    target_points_interp.append(
                        jnp.interp(t_vals, original_t, target_points[:, i])
                    )
                target_points = jnp.array(target_points_interp).T
        
        return target_points

    def _merge_pgf_params(self, custom_pgf_params):
        """
        Merges custom PGF parameters with defaults to ensure all required keys exist.
        
        Args:
            custom_pgf_params (dict): Custom PGF parameters
            
        Returns:
            dict: Complete PGF parameters with all required keys
        """
        # Start with defaults
        pgf_params = barqtools.get_default_pgf_params_dict()
        
        # Update with custom parameters
        if custom_pgf_params is not None:
            pgf_params.update(custom_pgf_params)
        
        return pgf_params

    def _fit_curve_to_control_points(self, target_curve=None, target_points=None, 
                                   n_samples=100, optimizer=None, max_iter=500,
                                   seed=None, pgf_params=None, prs_params=None):
        """
        Fits BARQ free control points to target curve using existing optimization infrastructure.
        
        This method leverages the prepare_optimization_loss + optimize pattern to fit 
        free_points while preserving BARQ structure and respecting custom PGF/PRS parameters.
        
        Args:
            target_curve (callable, optional): Function t -> [x,y,z] for t in [0,1]
            target_points (array, optional): Array of target points (n_points, 3)
            n_samples (int): Number of evaluation points for fitting. Default 100.
            optimizer (optax.GradientTransformation, optional): Optax optimizer.
                Default is optax.adam(0.01).
            max_iter (int): Maximum optimization iterations. Default 500.
            seed (int, optional): Random seed for initialization. Default None.
            pgf_params (dict, optional): Custom PGF parameters. If None, uses defaults.
            prs_params (dict, optional): Custom PRS parameters. If None, uses empty dict.
        
        Returns:
            jnp.array: Optimized free_points with shape (n_free_points, 3)
            
        Raises:
            ValueError: If neither target_curve nor target_points provided, or if
                       target_curve returns invalid values (NaN/Inf).
                       
        Examples:
            # Fit to analytical curve
            free_points = barq._fit_curve_to_control_points(
                target_curve=lambda t: [jnp.cos(2*jnp.pi*t), jnp.sin(2*jnp.pi*t), 0],
                optimizer=optax.adam(0.005),
                max_iter=1000
            )
            
            # Fit to discrete points  
            target_pts = jnp.array([[1,0,0], [0,1,0], [-1,0,0], [0,-1,0]])
            free_points = barq._fit_curve_to_control_points(
                target_points=target_pts,
                seed=42
            )
        
        Note:
            This method temporarily modifies self.params during optimization but 
            does not affect the permanent curve parameters. It uses MSE loss 
            between evaluated BARQ curve and target points.
            
            The method respects BARQ structure: fitted curves will always start/end 
            at origin and maintain gate-fixing constraints via PGF parameters.
        """
        # 1. Validation of inputs
        if target_curve is None and target_points is None:
            raise ValueError("Either target_curve or target_points must be provided")
        
        # 2. Set up parameters with proper merging
        merged_pgf_params = self._merge_pgf_params(pgf_params)
        if prs_params is None:
            prs_params = {}
        
        # 3. Generate target points using optimization grid (OPT_POINTS)
        # This ensures compatibility with prepare_optimization_loss infrastructure
        n_opt_points = settings.options['OPT_POINTS']
        target_points_opt = self._generate_target_points_for_optimization(
            target_curve, target_points, n_opt_points)
        
        # 4. Create MSE loss function compatible with frenet_dict infrastructure
        def fitting_loss_fn(frenet_dict):
            """
            MSE loss function that works with frenet_dict infrastructure.
            Uses frenet_dict['curve'] which contains curve points evaluated at optimization grid.
            """
            # Use curve points directly from frenet_dict
            # These are evaluated at the same points as target_points_opt
            curve_points = frenet_dict['curve']
            
            # MSE between evaluated curve and target points
            mse = jnp.mean(jnp.sum((curve_points - target_points_opt)**2, axis=1))
            return mse
        
        # 5. Store current parameters to restore later
        original_params = self.params
        
        # 6. Create temporary parameters for fitting with proper random initialization
        if seed is None:
            seed = 0
        rng = numpy.random.default_rng(seed)
        initial_free_points = rng.standard_normal((self.n_free_points, 3))
        initial_free_points = initial_free_points / numpy.linalg.norm(
            initial_free_points, axis=1, keepdims=True
        )
        initial_free_points = jnp.array(initial_free_points)
        
        temp_params = {
            'free_points': initial_free_points,
            'pgf_params': merged_pgf_params,
            'prs_params': prs_params
        }
        self.set_params(temp_params)
        
        # 7. Use existing optimization infrastructure
        self.prepare_optimization_loss([fitting_loss_fn, 1.0])
        
        # 8. Optimize using infrastructure with custom optimizer
        if optimizer is None:
            optimizer = optax.adam(0.01)
        self.optimize(optimizer=optimizer, max_iter=max_iter)
        
        # 9. Extract optimized free_points
        optimized_free_points = self.params['free_points']
        
        # 10. Restore original parameters
        if original_params is not None:
            self.set_params(original_params)
        
        return optimized_free_points

    def initialize_parameters(self, *, init_free_points=None,
                              init_pgf_params=None,
                              init_prs_params=None, seed=None,
                              fit_target_curve=None,
                              fit_target_points=None,
                              fit_n_samples=100,
                              fit_optimizer=None,
                              fit_max_iter=500,
                              fit_seed=None):

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
            fit_optimizer (optax.GradientTransformation, optional): Optimizer for 
                curve fitting. Default is optax.adam(0.01).
            fit_max_iter (int, optional): Maximum iterations for curve fitting.
                Default is 500.
            fit_seed (int, optional): Random seed specifically for curve fitting 
                initialization. If None, uses seed parameter.

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

        # Set up PGF and PRS parameters with proper merging
        merged_pgf_params = self._merge_pgf_params(init_pgf_params)
        if init_prs_params is None:
            init_prs_params = {}

        # Curve fitting initialization
        if curve_fitting_requested:
            init_free_points = self._fit_curve_to_control_points(
                target_curve=fit_target_curve,
                target_points=fit_target_points,
                n_samples=fit_n_samples,
                optimizer=fit_optimizer,
                max_iter=fit_max_iter,
                seed=fit_seed or seed,
                pgf_params=merged_pgf_params,
                prs_params=init_prs_params
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

        params['free_points'] = init_free_points
        params['pgf_params'] = merged_pgf_params
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