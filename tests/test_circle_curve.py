import jax.numpy as jnp
import matplotlib.pyplot as plt
from qurveros.spacecurve import SpaceCurve

# Define the circle function (unit circle in XY-plane)
def circle(x, params):
    return [jnp.cos(x), jnp.sin(x), 0]

# Create the original SpaceCurve instance
spacecurve_original = SpaceCurve(
    curve=circle,
    order=0,
    interval=[0, 2 * jnp.pi],
    params=None
)

# Evaluate the Frenet dictionary for the original curve
spacecurve_original.evaluate_frenet_dict(n_points=4096)

# Evaluate the control dictionary using 'XY' control mode
spacecurve_original.evaluate_control_dict('XY', n_points=4096)

# Get the control dictionary and initial frame for reconstruction
control_dict = spacecurve_original.get_control_dict()
original_frame = spacecurve_original.get_frenet_dict()['frame'][0]  # Frame at t=0
T0 = original_frame[0]  # Tangent
N0 = original_frame[1]  # Normal
B0 = original_frame[2]  # Binormal
# Construct initial rotation matrix: [-B, N, T] transposed
initial_rotation = jnp.stack([-B0, N0, T0], axis=0).T

# Reconstruct the SpaceCurve from the control dictionary
spacecurve_reconstructed = SpaceCurve(
    control_dict=control_dict,
    initial_rotation=initial_rotation
)

# Evaluate the Frenet dictionary for the reconstructed curve
spacecurve_reconstructed.evaluate_frenet_dict(n_points=4096)

# Generate control dictionary for the reconstructed curve
spacecurve_reconstructed.evaluate_control_dict('XY', n_points=4096)

# Plot and compare the original and reconstructed curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Original curve
original_curve = spacecurve_original.get_frenet_dict()['curve']
ax1.plot(original_curve[:, 0], original_curve[:, 1], label='Original Circle')
ax1.set_title('Original Curve')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.axis('equal')

# Reconstructed curve
reconstructed_curve = spacecurve_reconstructed.get_frenet_dict()['curve']
ax2.plot(reconstructed_curve[:, 0], reconstructed_curve[:, 1], label='Reconstructed Circle', color='orange')
ax2.set_title('Reconstructed Curve')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()
ax2.axis('equal')

plt.tight_layout()
plt.show()

# Numerical comparison: Print start and end points
print("Original curve start:", original_curve[0])
print("Original curve end:", original_curve[-1])
print("Reconstructed curve start:", reconstructed_curve[0])
print("Reconstructed curve end:", reconstructed_curve[-1])

# Evaluate and print robustness properties
spacecurve_original.evaluate_robustness_properties()
spacecurve_reconstructed.evaluate_robustness_properties()
original_props = spacecurve_original.get_robustness_properties()
reconstructed_props = spacecurve_reconstructed.get_robustness_properties()
print("Original Robustness Properties:", original_props)
print("Reconstructed Robustness Properties:", reconstructed_props)