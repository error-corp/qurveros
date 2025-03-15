This file outlines some known issues of `qurveros` and potential solutions.

# High-bandwidth control fields

Depending on the choice of the ansatz, some resulting control fields might
produce significantly lower fidelity than the expected one. 

The user can increase the number of the traversal parameter samples using a file
named `qurveros_settings.json`. The package searches recursively up to 3 times
to find if such a configuration file exists and overrides the default parameters
with the parameters described in the file.
The number of simulation points can be also set in a similar way.

The settings can be found using:
```python
from qurveros.settings import settings
print(settings)
```

Most of the values are used in runtime hence can be updated 
using the `options` dictionary attribute.

# Floating point errors

There are cases that floating point errors may appear.

During the optimization phase, floating point
errors typically emerge in the boundaries of the evolution.
This problem occurred when optimizing in BARQ. 
If the parameter interval is set to operate in `(0,1)`, 
the inflection points at the boundaries are excluded, which was found to be 
a working solution. 

The optimization can still produce robust pulses, since
most of the constraints either enter in an integral form where the boundary 
points do not contribute significantly or the local quantities to minimize
are situated inside the interval.

Inflection points are detected when a derivative of the curve is nearly 
parallel to the tangent (in the non-arclength parametrization).
Hence floating point errors may occur when the norm of the cross-product 
is small.

When the `prepare_optimization_loss` method is invoked, an optional `interval`
argument can be provided that dictates the interval of the traversal parameter
used in the optimization.

# Data types
When a SpaceCurve instance is created, the curve function is internally 
wrapped to return floating point arrays. The auxiliary parameters 
are not checked against their data type, hence some curve functions may cause 
the calculation of the frenet dictionary to throw an exception, if integers
are used.