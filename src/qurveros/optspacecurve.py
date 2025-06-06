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
from tqdm import tqdm

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
        self.last_valid_params = None
        self.last_valid_loss = None
        self.is_closed = False  # Default: not closed
        self.evaluate = lambda x, params=None: self.curve(x, params)  # Default evaluate method

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
            self.last_valid_params = init_params
            self.last_valid_loss = self.opt_loss(init_params) if self.opt_loss else None

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

    def _validate_params(self, params):
        """Validate parameters for numerical stability and correctness.
        
        Args:
            params: Parameters to validate (can be array or dict)
            
        Returns:
            tuple: (is_valid, message)
        """
        # Check if loss function is set
        if self.opt_loss is None:
            return False, "Loss function not set"
        
        try:
            # Only check for NaN/Inf recursively, not loss/gradient
            def check_nan_inf(p):
                if isinstance(p, dict):
                    for key, value in p.items():
                        if isinstance(value, (jnp.ndarray, numpy.ndarray)):
                            if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                                return False, f"NaN or Inf values found in {key}"
                        elif isinstance(value, dict):
                            is_valid, msg = check_nan_inf(value)
                            if not is_valid:
                                return False, f"Invalid {key}: {msg}"
                else:
                    if jnp.any(jnp.isnan(p)) or jnp.any(jnp.isinf(p)):
                        return False, "NaN or Inf values found in parameters"
                return True, "OK"
            is_valid, msg = check_nan_inf(params)
            if not is_valid:
                return False, msg
            # Only call loss/gradient on the full parameter set
            try:
                loss = self.opt_loss(params)
                if jnp.isnan(loss) or jnp.isinf(loss):
                    return False, f"Loss evaluation resulted in {loss}"
            except Exception as e:
                return False, f"Loss evaluation failed: {str(e)}"
            try:
                grad = self.loss_grad(params) if hasattr(self, 'loss_grad') and callable(self.loss_grad) else jax.grad(self.opt_loss)(params)
                if isinstance(grad, dict):
                    for key, value in grad.items():
                        if isinstance(value, (jnp.ndarray, numpy.ndarray)):
                            if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                                return False, f"NaN or Inf values in gradient for {key}"
                else:
                    if jnp.any(jnp.isnan(grad)) or jnp.any(jnp.isinf(grad)):
                        return False, "NaN or Inf values in gradient"
            except Exception as e:
                return False, f"Gradient evaluation failed: {str(e)}"
            # BARQ-specific checks
            if isinstance(self, BarqCurve):
                if 'pgf_params' in params:
                    pgf = params['pgf_params']
                    if not all(v > 0 for v in pgf.values() if isinstance(v, (int, float))):
                        return False, "PGF parameters must be positive"
                if 'free_points' in params:
                    points = params['free_points']
                    norms = jnp.linalg.norm(points, axis=1)
                    if jnp.any(norms < 1e-6):
                        return False, "Free points have zero or near-zero norms"
            # Check boundary conditions
            if hasattr(self, 'is_closed') and self.is_closed:
                try:
                    if hasattr(self, 'evaluate') and callable(self.evaluate):
                        start = self.evaluate(0.0, params)
                        end = self.evaluate(1.0, params)
                        if not jnp.allclose(start, end, rtol=1e-5, atol=1e-5):
                            return False, "Curve is not closed"
                except Exception as e:
                    return False, f"Boundary condition check failed: {str(e)}"
            if hasattr(self, 'check_curvature') and self.check_curvature is not None:
                try:
                    if not self.check_curvature(params):
                        return False, "Curvature check failed"
                except Exception as e:
                    return False, f"Curvature check failed: {str(e)}"
            return True, "Parameters are valid"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def _validate_boundary_conditions(self, params):
        """
        Validates boundary conditions for the curve.
        
        Args:
            params: The parameters to validate
            
        Returns:
            bool: True if boundary conditions are satisfied
        """
        try:
            # Get curve values at boundaries
            x_values = jnp.array([self.interval[0], self.interval[1]])
            curve_values = self._frenet_dict_fun(x_values, params)
            
            # Check if curve is closed
            if not jnp.allclose(curve_values['curve'][0], curve_values['curve'][-1], atol=1e-6):
                return False
                
            # Check if curvature vanishes at boundaries
            if not jnp.allclose(curve_values['curvature'][0], 0, atol=1e-6) or \
               not jnp.allclose(curve_values['curvature'][-1], 0, atol=1e-6):
                return False
                
            return True
        except:
            return False

    def _perturb_params(self, params, scale=1e-4, strategy='random', error_magnitude=1.0):
        """Perturb parameters to recover from numerical instability.

        Args:
            params: Parameters to perturb
            scale: Scale of perturbation
            strategy: Perturbation strategy ('random', 'adaptive', 'gradient', 'trust_region', 'line_search')
            error_magnitude: Magnitude of the error that triggered perturbation
            
        Returns:
            Perturbed parameters
        """
        if strategy == 'random':
            if isinstance(params, dict):
                perturbed = {}
                for key, value in params.items():
                    if isinstance(value, (jnp.ndarray, numpy.ndarray)):
                        noise = jax.random.normal(jax.random.PRNGKey(0), value.shape) * scale
                        perturbed[key] = value + noise
                    else:
                        perturbed[key] = value
                return perturbed
            else:
                noise = jax.random.normal(jax.random.PRNGKey(0), params.shape) * scale
                return params + noise
            
        elif strategy == 'adaptive':
            # Scale perturbation based on error magnitude
            adaptive_scale = scale / (1.0 + error_magnitude)
            return self._perturb_params(params, scale=adaptive_scale, strategy='random')
            
        elif strategy == 'gradient':
            try:
                if isinstance(params, dict):
                    grads = jax.grad(self.opt_loss)(params)
                    perturbed = {}
                    for key, value in params.items():
                        if isinstance(value, (jnp.ndarray, numpy.ndarray)):
                            grad = grads[key]
                            perturbed[key] = value - scale * grad
                        else:
                            perturbed[key] = value
                    return perturbed
                else:
                    grad = jax.grad(self.opt_loss)(params)
                    return params - scale * grad
            except Exception:
                # Fall back to random perturbation if gradient computation fails
                return self._perturb_params(params, scale=scale, strategy='random')
            
        elif strategy == 'trust_region':
            try:
                if isinstance(params, dict):
                    grads = jax.grad(self.opt_loss)(params)
                    hessian = jax.jacfwd(jax.grad(self.opt_loss))(params)
                    perturbed = {}
                    for key, value in params.items():
                        if isinstance(value, (jnp.ndarray, numpy.ndarray)):
                            grad = grads[key]
                            hess = hessian[key]
                            # Simple trust region update
                            update = jnp.linalg.solve(hess + scale * jnp.eye(hess.shape[0]), grad)
                            perturbed[key] = value - update
                        else:
                            perturbed[key] = value
                    return perturbed
                else:
                    grad = jax.grad(self.opt_loss)(params)
                    hessian = jax.jacfwd(jax.grad(self.opt_loss))(params)
                    update = jnp.linalg.solve(hessian + scale * jnp.eye(hessian.shape[0]), grad)
                    return params - update
            except Exception:
                return self._perturb_params(params, scale=scale, strategy='random')
            
        elif strategy == 'line_search':
            try:
                if isinstance(params, dict):
                    grads = jax.grad(self.opt_loss)(params)
                    perturbed = {}
                    for key, value in params.items():
                        if isinstance(value, (jnp.ndarray, numpy.ndarray)):
                            grad = grads[key]
                            # Backtracking line search
                            alpha = 1.0
                            while alpha > 1e-10:
                                new_value = value - alpha * grad
                                if self.opt_loss({**params, key: new_value}) < self.opt_loss(params):
                                    perturbed[key] = new_value
                                    break
                                alpha *= 0.5
                            else:
                                perturbed[key] = value
                        else:
                            perturbed[key] = value
                    return perturbed
                else:
                    grad = jax.grad(self.opt_loss)(params)
                    # Backtracking line search
                    alpha = 1.0
                    while alpha > 1e-10:
                        new_params = params - alpha * grad
                        if self.opt_loss(new_params) < self.opt_loss(params):
                            return new_params
                        alpha *= 0.5
                    return params
            except Exception:
                return self._perturb_params(params, scale=scale, strategy='random')
            
        else:
            raise ValueError(f"Unknown perturbation strategy: {strategy}")

    def optimize(self, optimizer=None, max_iter=1000, recovery_config=None, verbose=False):
        """Optimize the curve's auxiliary parameters.
        
        Args:
            optimizer: Optax optimizer to use (default: Adam with lr=0.01)
            max_iter: Maximum number of optimization iterations
            recovery_config: Configuration for recovery from numerical instability
            verbose: Whether to print detailed progress information
            
        Returns:
            dict: Optimization statistics
        """
        if self.params is None:
            raise RuntimeError("Parameters not set. Call initialize_parameters() first.")
        
        # Default recovery configuration
        if recovery_config is None:
            recovery_config = {
                'max_retries': 3,
                'perturbation_scale': 1e-4,
                'perturbation_strategy': 'adaptive',
                'validation_thresholds': {
                    'nan_threshold': 1e-10,
                    'inf_threshold': 1e10,
                    'gradient_threshold': 1e6,
                    'loss_threshold': 1e10
                }
            }
        
        # Initialize optimizer
        if optimizer is None:
            optimizer = optax.adam(learning_rate=0.01)
        
        # Initialize optimization state
        opt_state = optimizer.init(self.params)
        best_params = self.params
        best_loss = float('inf')
        last_valid_params = self.params
        last_valid_loss = float('inf')
        
        # Initialize statistics
        stats = {
            'iterations': 0,
            'retries': 0,
            'best_loss': float('inf'),
            'loss_history': [],
            'gradient_norms': [],
            'validation_errors': [],
            'recovery_attempts': []
        }
        
        # JIT-compiled step function
        @jax.jit
        def step(params, opt_state):
            loss_val, grads = jax.value_and_grad(self.opt_loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val, grads
        
        # Main optimization loop
        with tqdm(total=max_iter, desc="Optimizing") as pbar:
            for i in range(max_iter):
                try:
                    # Optimization step
                    self.params, opt_state, loss_val, grads = step(self.params, opt_state)
                    
                    # Update statistics
                    stats['iterations'] = i + 1
                    stats['loss_history'].append(float(loss_val))
                    grad_norm = jnp.linalg.norm(jax.tree_util.tree_leaves(grads)[0])
                    stats['gradient_norms'].append(float(grad_norm))
                    
                    # Validate parameters
                    is_valid, error_msg = self._validate_params(self.params)
                    
                    if is_valid:
                        # Update best parameters
                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_params = self.params
                            stats['best_loss'] = float(best_loss)
                        
                        # Update last valid parameters
                        last_valid_params = self.params
                        last_valid_loss = loss_val
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss_val:.2e}",
                            'grad_norm': f"{grad_norm:.2e}"
                        })
                    else:
                        # Handle invalid parameters
                        stats['validation_errors'].append({
                            'iteration': i,
                            'error': error_msg,
                            'loss': float(loss_val)
                        })
                        
                        # Attempt recovery
                        retry_count = 0
                        while retry_count < recovery_config['max_retries']:
                            stats['retries'] += 1
                            stats['recovery_attempts'].append({
                                'iteration': i,
                                'attempt': retry_count + 1,
                                'strategy': recovery_config['perturbation_strategy']
                            })
                            
                            # Perturb parameters
                            self.params = self._perturb_params(
                                last_valid_params,
                                recovery_config['perturbation_scale'],
                                recovery_config['perturbation_strategy'],
                                error_magnitude=loss_val - last_valid_loss
                            )
                            
                            # Validate perturbed parameters
                            is_valid, error_msg = self._validate_params(self.params)
                            if is_valid:
                                break
                            
                            retry_count += 1
                        
                        if retry_count >= recovery_config['max_retries']:
                            if verbose:
                                print(f"\nRecovery failed after {recovery_config['max_retries']} attempts")
                                print(f"Last error: {error_msg}")
                            # Restore best parameters
                            self.params = best_params
                            break
                    
                    pbar.update(1)
                    
                except Exception as e:
                    if verbose:
                        print(f"\nError during optimization: {str(e)}")
                    stats['validation_errors'].append({
                        'iteration': i,
                        'error': str(e),
                        'loss': float('inf')
                    })
                    # Restore best parameters
                    self.params = best_params
                    break
        
        # Final validation
        is_valid, error_msg = self._validate_params(self.params)
        if not is_valid and verbose:
            print(f"\nFinal parameters invalid: {error_msg}")
            print("Restoring best valid parameters")
            self.params = best_params
        
        return stats

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

    def initialize_parameters(self, *, init_free_points=None,
                              init_pgf_params=None,
                              init_prs_params=None, seed=None):

        """
        Initializes the parameters for the BARQ method.
        The parameters are passed to the curve as a dictionary with entries
        containing the free points, the pgf parameters and the prs parameters.

        If no initial free points are provided, a total of n_free_points random
        points are drawn and normalized to unit magnitude.

        Raises:
            ValueError: If the number of free points provided upon
            instantiation do not agree with the dimensions of the initial
            points.
        """

        params = {}

        if seed is None:
            seed = 0

        rng = numpy.random.default_rng(seed)

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
