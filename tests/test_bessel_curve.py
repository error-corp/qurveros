import jax.numpy as jnp
import matplotlib.pyplot as plt
from qurveros.spacecurve import SpaceCurve
import qutip
from qurveros.qubit_bench import simulator
from qurveros import frametools

# Define the tangent vector function for the curve
def curve(x, params):
    theta = params[0] * jnp.cos(x)
    phi = params[1] * theta
    x_comp = jnp.cos(phi) * jnp.sin(theta)
    y_comp = jnp.sin(phi) * jnp.sin(theta)
    z_comp = jnp.cos(theta)
    return [x_comp, y_comp, z_comp]

# Set parameters
params = [5.5201, 0.5660]

# Create the original SpaceCurve instance
spacecurve_original = SpaceCurve(
    curve=curve,
    order=1,  # Tangent vector
    interval=[0, 2 * jnp.pi],
    params=params
)

# Evaluate the Frenet dictionary for the original curve
spacecurve_original.evaluate_frenet_dict(n_points=500)

# Generate the control dictionary using 'XY' control mode
spacecurve_original.evaluate_control_dict('XY', n_points=500)

# Extract data for reconstruction
control_dict = spacecurve_original.get_control_dict()
original_frame = spacecurve_original.get_frenet_dict()['frame'][0]
T0, N0, B0 = original_frame[0], original_frame[1], original_frame[2]  # Tangent, Normal, Binormal
initial_rotation = jnp.stack([-B0, N0, T0], axis=0).T  # Initial rotation matrix

# Reconstruct the SpaceCurve
spacecurve_reconstructed = SpaceCurve(
    control_dict=control_dict,
    initial_rotation=initial_rotation
)

# Evaluate the Frenet dictionary for the reconstructed curve
spacecurve_reconstructed.evaluate_frenet_dict(n_points=500)

#Generate control dictionary for the reconstructed curve
spacecurve_reconstructed.evaluate_control_dict('XY', n_points=500)

# Plot Original Curve in Figure 1
fig_original = plt.figure(figsize=(12, 8))

# 3D plot
ax1 = fig_original.add_subplot(221, projection='3d')
original_curve = spacecurve_original.get_frenet_dict()['curve']
ax1.plot(original_curve[:, 0], original_curve[:, 1], original_curve[:, 2], 
         label='Original Curve', color='blue')
ax1.set_title('Original Curve (3D)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# XY projection
ax2 = fig_original.add_subplot(222)
ax2.plot(original_curve[:, 0], original_curve[:, 1], 
         label='Original XY', color='blue')
ax2.set_title('Original XY Projection')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.grid(True)
ax2.legend()

# XZ projection
ax3 = fig_original.add_subplot(223)
ax3.plot(original_curve[:, 0], original_curve[:, 2], 
         label='Original XZ', color='blue')
ax3.set_title('Original XZ Projection')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.axis('equal')
ax3.grid(True)
ax3.legend()

# YZ projection
ax4 = fig_original.add_subplot(224)
ax4.plot(original_curve[:, 1], original_curve[:, 2], 
         label='Original YZ', color='blue')
ax4.set_title('Original YZ Projection')
ax4.set_xlabel('Y')
ax4.set_ylabel('Z')
ax4.axis('equal')
ax4.grid(True)
ax4.legend()

plt.tight_layout()

# Plot Reconstructed Curve in Figure 2
fig_reconstructed = plt.figure(figsize=(12, 8))

# 3D plot
ax5 = fig_reconstructed.add_subplot(221, projection='3d')
reconstructed_curve = spacecurve_reconstructed.get_frenet_dict()['curve']
ax5.plot(reconstructed_curve[:, 0], reconstructed_curve[:, 1], reconstructed_curve[:, 2], 
         label='Reconstructed Curve', color='orange')
ax5.set_title('Reconstructed Curve (3D)')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.legend()

# XY projection
ax6 = fig_reconstructed.add_subplot(222)
ax6.plot(reconstructed_curve[:, 0], reconstructed_curve[:, 1], 
         label='Reconstructed XY', color='orange')
ax6.set_title('Reconstructed XY Projection')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.axis('equal')
ax6.grid(True)
ax6.legend()

# XZ projection
ax7 = fig_reconstructed.add_subplot(223)
ax7.plot(reconstructed_curve[:, 0], reconstructed_curve[:, 2], 
         label='Reconstructed XZ', color='orange')
ax7.set_title('Reconstructed XZ Projection')
ax7.set_xlabel('X')
ax7.set_ylabel('Z')
ax7.axis('equal')
ax7.grid(True)
ax7.legend()

# YZ projection
ax8 = fig_reconstructed.add_subplot(224)
ax8.plot(reconstructed_curve[:, 1], reconstructed_curve[:, 2], 
         label='Reconstructed YZ', color='orange')
ax8.set_title('Reconstructed YZ Projection')
ax8.set_xlabel('Y')
ax8.set_ylabel('Z')
ax8.axis('equal')
ax8.grid(True)
ax8.legend()

plt.tight_layout()

# Display both figures
plt.show()

# Compare Robustness Properties
spacecurve_original.evaluate_robustness_properties()
spacecurve_reconstructed.evaluate_robustness_properties()
original_props = spacecurve_original.get_robustness_properties()
reconstructed_props = spacecurve_reconstructed.get_robustness_properties()
print("Original Robustness Properties:", original_props)
print("Reconstructed Robustness Properties:", reconstructed_props)

# Quantum Simulations and Fidelity Checks
u_target = qutip.Qobj(jnp.eye(2))  # Target unitary (identity for simplicity)

# Simulate original control dictionary
sim_dict_original = simulator.simulate_control_dict(spacecurve_original.get_control_dict(), u_target)
print("Original Avg Gate Fidelity:", sim_dict_original['avg_gate_fidelity'])

# Simulate reconstructed control dictionary
sim_dict_reconstructed = simulator.simulate_control_dict(spacecurve_reconstructed.get_control_dict(), u_target)
print("Reconstructed Avg Gate Fidelity:", sim_dict_reconstructed['avg_gate_fidelity'])

# Fidelity checks
adj_fidelity_original = frametools.calculate_adj_fidelity(sim_dict_original['adj_final'], sim_dict_original['adj_target'])
adj_fidelity_reconstructed = frametools.calculate_adj_fidelity(sim_dict_reconstructed['adj_final'], sim_dict_reconstructed['adj_target'])
print("Original Adjoint Fidelity:", adj_fidelity_original)
print("Reconstructed Adjoint Fidelity:", adj_fidelity_reconstructed)