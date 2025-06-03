import jax
import jax.numpy as jnp
import numpy as np
from qurveros.optspacecurve import BarqCurve
from qurveros.optspacecurve import OptimizableSpaceCurve
from qurveros.optspacecurve import SpaceCurve

import warnings

def test_dtype_conversion():
    """Test that all numeric parameters are properly converted to JAX float types"""
    # Capture warnings to verify conversions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test 1: Direct parameter setting in SpaceCurve
        sc = SpaceCurve(curve=lambda x, p: x*p, order=0, interval=[0,1])
        
        # Create test params with mixed types
        test_params = {
            'int_scalar': 42,
            'float_scalar': 3.14,
            'np_int_array': np.array([1, 2, 3], dtype=np.int32),
            'np_float_array': np.array([1.1, 2.2], dtype=np.float32),
            'nested': {
                'int_list': [4, 5, 6],
                'jax_array': jnp.array([7, 8], dtype=jnp.int32)
            }
        }
        
        sc.set_params(test_params)
        params = sc.get_params()
        
        # Verify conversions
        assert isinstance(params['int_scalar'], jnp.ndarray)
        assert params['int_scalar'].dtype in (jnp.float32, jnp.float64)
        
        assert isinstance(params['np_int_array'], jnp.ndarray)
        assert params['np_int_array'].dtype in (jnp.float32, jnp.float64)
        
        assert isinstance(params['nested']['int_list'], jnp.ndarray)
        assert params['nested']['int_list'].dtype in (jnp.float32, jnp.float64)
        
        # Even pre-existing JAX int arrays should be converted
        assert params['nested']['jax_array'].dtype in (jnp.float32, jnp.float64)
        
        # Test 2: BarqCurve initialization path
        adj_target = jnp.eye(3)
        barq = BarqCurve(adj_target=adj_target, n_free_points=3)
        
        barq.initialize_parameters(
            init_free_points=np.array([[1,2,3],[4,5,6]], dtype=np.int32),
            init_prs_params={'test_int': 42}
        )
        
        barq_params = barq.get_params()
        assert barq_params['free_points'].dtype in (jnp.float32, jnp.float64)
        assert barq_params['prs_params']['test_int'].dtype in (jnp.float32, jnp.float64)
        
        # Verify warnings were issued for conversions
        assert len(w) > 0, "Expected conversion warnings"
        print("\nConversion warnings detected (expected):")
        for warning in w:
            print(f"- {warning.message}")
        
        print("\nAll dtype conversions passed successfully!")

if __name__ == "__main__":
    # Configure JAX to use float32 (typical default)
    jax.config.update('jax_enable_x64', False)
    print("Testing with jax_enable_x64=False (float32 mode)")
    test_dtype_conversion()
    
    # Retest with float64 mode
    jax.config.update('jax_enable_x64', True)
    print("\nTesting with jax_enable_x64=True (float64 mode)")
    test_dtype_conversion()