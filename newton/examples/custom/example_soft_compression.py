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
# Rigid Sphere Falling onto Soft Box
#
# A rigid sphere falls and impacts a soft deformable box on the ground
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
        self.sim_substeps = 32  # Number of physics substeps per frame
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
        builder.default_particle_radius = 0.001
        
        # Setup grid parameters for soft box
        cell_dim = 3  # 3x3x3 cells
        cell_size = 0.05  # 5cm cells
        
        # Compute particle density for 1.0 kg total mass
        total_mass = 1.0
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
            fix_bottom=True,
        )
        
        # Add rigid sphere falling from above
        sphere_mass = 0.5  # kg
        sphere_radius = 0.05  # 5cm radius
        sphere_height = 0.5  # Starting height: 50cm above ground
        
        # Create sphere body with exact mass
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 0.0  # Mass comes from add_body, not shape
        
        body_sphere = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, sphere_height), q=wp.quat_identity()),
            mass=sphere_mass
        )
        
        builder.add_shape_sphere(
            body=body_sphere,
            radius=sphere_radius,
            cfg=sphere_cfg,
        )

        # Finalize model
        self.model = builder.finalize()
        
        # Contact properties
        # STABILITY: Higher contact stiffness and damping prevent interpenetration
        self.model.soft_contact_ke = 1.0e4  # Increased from 1.0e2 (stiffer contacts)
        self.model.soft_contact_kd = 1.0e2  # Increased from 1.0e0 (more contact damping)
        self.model.soft_contact_mu = 0.5    # Reduced from 1.0 (lower friction can help)
       
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
        
        # Store sphere parameters for force tracking
        self.sphere_mass = sphere_mass
        self.sphere_body_id = 0  # First (and only) rigid body
        self.frame_counter = 0
        
        # Print info
        print(f"\n{'='*70}")
        print(f"Rigid Sphere Falling onto Soft Box")
        print(f"{'='*70}")
        print(f"Soft box:")
        print(f"  Size: {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm")
        print(f"  Particles: {self.model.particle_count}")
        print(f"  Tetrahedra: {self.model.tet_count}")
        print(f"  Mass: {total_mass:.3f} kg")
        print(f"Rigid sphere:")
        print(f"  Radius: {sphere_radius*100:.0f}cm")
        print(f"  Mass: {sphere_mass:.3f} kg")
        print(f"  Drop height: {sphere_height*100:.0f}cm")
        print(f"  Bodies: {self.model.body_count}")
        print(f"  Expected weight: {sphere_mass * 9.81:.2f} N")
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
        """Print contact forces on sphere"""
        # Get body forces (from contacts with soft box and ground)
        body_forces = self.state_0.body_f.numpy()
        
        # Sphere force
        sphere_force = body_forces[self.sphere_body_id, 0:3]
        sphere_torque = body_forces[self.sphere_body_id, 3:6]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        # Get sphere position and velocity
        body_q = self.state_0.body_q.numpy()
        sphere_pos = body_q[self.sphere_body_id, 0:3]
        sphere_z = sphere_pos[2]
        
        body_qd = self.state_0.body_qd.numpy()
        sphere_vel = body_qd[self.sphere_body_id, 0:3]
        sphere_vz = sphere_vel[2]
        
        # Expected weight force
        expected_weight = self.sphere_mass * 9.81
        
        print(f"Frame {self.frame_counter:4d} | t={self.sim_time:.2f}s | "
              f"Sphere z={sphere_z*100:.2f}cm, vz={sphere_vz:.3f}m/s")
        print(f"  Contact force: F=[{sphere_force[0]:7.2f}, {sphere_force[1]:7.2f}, {sphere_force[2]:7.2f}] N  "
              f"|F|={sphere_force_mag:7.2f} N")
        print(f"  Expected weight: {expected_weight:.2f} N")
        
        # Show if sphere is in contact (force magnitude > small threshold)
        if sphere_force_mag > 0.0001:
            force_ratio = sphere_force_mag / expected_weight
            print(f"  Force/Weight ratio: {force_ratio:.2f}x")
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
