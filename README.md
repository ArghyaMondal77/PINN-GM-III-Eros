\# PINN-Asteroid-Gravity: 433 Eros Landing Model



A fast, high-fidelity Physics-Informed Neural Network (PINN) built in \*\*JAX/Flax\*\* to map the chaotic gravitational field of Asteroid 433 Eros. 



This model acts as a highly optimized surrogate for computationally expensive C++ Polyhedron gravity engines, maintaining \*\*>99% accuracy\*\* while drastically reducing inference time for real-time aerospace applications (e.g., terminal descent and landing simulations).



\## The Engineering Challenge

Standard point-mass gravity equations ($GM/r^2$) work perfectly in deep space, but completely fail within 5 kilometers of an asteroid's surface. In the "Danger Zone" near the surface, the gravity field is chaotic, dictated by localized craters, ridges, and valleys. 



Traditional Polyhedron integration solves this but is too computationally heavy for real-time onboard flight computers. This project compresses the physics of Laplace's equation into a differentiable neural network.



\## Methodology \& Architecture

\* \*\*Framework:\*\* JAX / Flax / Optax

\* \*\*Auto-Diff Physics:\*\* The network outputs a scalar gravitational potential ($\\hat{U}$). The spatial acceleration vector is derived instantly via exact automatic differentiation: $a = -\\nabla \\hat{U}$.

\* \*\*Tiered Conformal Sampling:\*\* To prevent volume dilution and ensure the model learns complex crater physics, the 50,000-point training dataset was generated using surface-normal offsets:

&#x20; \* \*\*Tier 1 (50m - 1km):\*\* 50% of data (The Danger Zone)

&#x20; \* \*\*Tier 2 (1km - 5km):\*\* 30% of data (Transition Zone)

&#x20; \* \*\*Tier 3 (5km - 15km):\*\* 20% of data (Mid-Field)

\* \*\*Mathematical Boundary Condition:\*\* Implemented a Heaviside transition function calibrated to the exact scaled gravitational parameter ($\\mu\_{scaled}$) of Eros to seamlessly hand off to deep-space physics at high altitudes.



\## Validation Metrics (Unseen Test Data)

The model was stress-tested against 15,000 completely unseen conformal coordinates evaluated by a strict C++ Polyhedron engine. 



| Altitude Tier | Mean Percent Error (MPE) | Root Mean Sq Error (RMSE) | Relative $L\_2$ Error |

| :--- | :--- | :--- | :--- |

| \*\*Tier 1 (50m - 1km)\*\* | 0.72% | 4.26e-5 m/s² | \*\*0.0090\*\* (0.9%) |

| \*\*Tier 2 (1km - 5km)\*\* | 0.46% | 1.62e-5 m/s² | \*\*0.0051\*\* (0.5%) |

| \*\*Tier 3 (5km - 15km)\*\* | 0.41% | 6.58e-6 m/s² | \*\*0.0045\*\* (0.4%) |



\*Note: A Relative $L\_2$ Error below 0.01 (1%) in the near-surface regime is considered flight-ready for surrogate landing models.\*



\## Interactive 3D Visualization

The notebook includes a real-time Plotly WebGL visualization that maps the predicted gravity vectors over the 3D mesh of Asteroid 433 Eros, allowing for point-by-point hover comparison between the PINN prediction and the Polyhedron ground truth.



\## Repository Structure

\* `/data/` - Contains the 3D `.obj` mesh and the generated numpy training arrays.

\* `/models/` - Contains the serialized `.msgpack` JAX weights for the trained landing model.

\* `/notebooks/` - Contains the heavily commented Jupyter notebook with data generation, training loops, and Plotly visualization.



\## Quick Start

```bash

\# Clone the repository

git clone \[https://github.com/ArghyaMondal77/PINN-Asteroid-Gravity.git](https://github.com/ArghyaMondal77/PINN-Asteroid-Gravity.git)

cd PINN-Asteroid-Gravity



\# Install dependencies

pip install jax flax optax trimesh plotly scipy

