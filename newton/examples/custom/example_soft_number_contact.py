# 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Soft Cube on Floor
#
# A soft deformable cube resting on the ground - shows contact forces
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, verbose=False):
        
        # Simulation parameters
        self.fps = 60  # Frames per second for visualization
        self.frame_dt = 1.0 / self.fps  # Time step per frame
        self.sim_time = 0.0
        
        # Physics substeps (higher = more stable but slower)
        self.sim_substeps = 64  # Number of physics substeps per frame
        self.sim_dt = self.frame_dt / self.sim_substeps  # Physics time step

        # Setup rendering
        self.viewer = viewer
        self.verbose = verbose
        
        print(f"Simulation configured:")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Frame dt: {self.frame_dt:.4f} s")
        print(f"  Physics substeps: {self.sim_substeps}")
        print(f"  Physics dt: {self.sim_dt:.6f} s")
        print()
        
        # Build model
        builder = newton.ModelBuilder()

        # Ground plane configuration
        # STABILITY: Match contact properties with soft contact settings
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = 1.0e4  # Increased for stiffer contacts
        ground_cfg.kd = 1.0e2  # Increased for more damping
        builder.add_ground_plane(cfg=ground_cfg)
        
        # Create FEM MODEL
        builder.default_particle_radius = 0.01
        
        # Setup grid parameters for soft box
        cell_dim = 3  # 3x3x3 cells
        cell_size = 0.05  # 5cm cells
        
        # Compute particle density for 1.0 kg total mass
        total_mass = 2.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
        if self.verbose:
            print(f"Particle density: {particle_density}")
        
        # Compute Lame parameters
        young_mod = 1.5e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        
        # STABILITY: Add material damping to dissipate energy
        k_damp = 100.0  # Increased from 0.0 (damping stabilizes high-frequency oscillations)
        
        # Add soft grid (box) on ground
        # Position: bottom of box at z=0
        box_height = cell_dim * cell_size
        
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * cell_size * cell_dim, -0.5 * cell_size * cell_dim, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,  # Material damping for stability
            tri_ke=1e-3,
            tri_ka=1e-3,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        # Finalize model
        self.model = builder.finalize()
        
        # Contact properties
        # STABILITY: Higher contact stiffness and damping prevent interpenetration
        self.model.soft_contact_ke = 1.0e4  # Increased from 1.0e2 (stiffer contacts)
        self.model.soft_contact_kd = 1.0e2  # Increased from 1.0e0 (more contact damping)
        self.model.soft_contact_mu = 5.0    # Reduced from 1.0 (lower friction can help)
       
        # Select the solver
        self.solver = newton.solvers.SolverSemiImplicit(model=self.model)

        # Allocate sim states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        # Rendering
        self.viewer.set_model(self.model)

        # Capture forward passes
        self.capture()
        
        # Frame counter for printing
        self.frame_counter = 0
        
        # Print info
        print(f"\n{'='*70}")
        print(f"Soft Cube on Floor")
        print(f"{'='*70}")
        print(f"Soft box:")
        print(f"  Size: {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm")
        print(f"  Particles: {self.model.particle_count}")
        print(f"  Tetrahedra: {self.model.tet_count}")
        print(f"  Mass: {total_mass:.3f} kg")
        print(f"  Expected weight: {total_mass * 9.81:.2f} N")
        print(f"{'='*70}\n")

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
           
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0 

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt
        self.frame_counter += 1
        
        # Print forces every 10 frames
        if self.frame_counter % 10 == 0:
            self.print_forces()
    
    def print_forces(self):
        """Print contact information for soft cube"""
        # Get soft cube particle forces (contact + internal)
        particle_forces = self.state_0.particle_f.numpy()
        
        # Count particles in contact with ground (z close to 0)
        particle_q = self.state_0.particle_q.numpy()
        particles_on_ground = np.sum(particle_q[:, 2] < 0.01)  # Within 1cm of ground
        
        # Count particles with significant forces
        particle_force_mags = np.linalg.norm(particle_forces, axis=1)
        particles_with_contact = np.sum(particle_force_mags > 1.0)  # Force > 1N threshold
        
        # Total force on soft cube particles
        total_particle_force = np.sum(particle_forces, axis=0)
        total_particle_force_mag = np.linalg.norm(total_particle_force)
        
        # Average position
        avg_z = np.mean(particle_q[:, 2])
        min_z = np.min(particle_q[:, 2])
        max_z = np.max(particle_q[:, 2])
        
        print(f"Frame {self.frame_counter:4d} | t={self.sim_time:.2f}s")
        print(f"  Soft cube position:")
        print(f"    Avg z={avg_z*100:.2f}cm, Min z={min_z*100:.2f}cm, Max z={max_z*100:.2f}cm")
        print(f"  Soft cube contacts:")
        print(f"    Particles on ground: {particles_on_ground}/{self.model.particle_count}")
        print(f"    Particles with forces >1N: {particles_with_contact}/{self.model.particle_count}")
        print(f"    Total force on cube: F=[{total_particle_force[0]:7.2f}, {total_particle_force[1]:7.2f}, {total_particle_force[2]:7.2f}] N")
        print(f"    Total force magnitude: |F|={total_particle_force_mag:.2f} N")
        print() 

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)
