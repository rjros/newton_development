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
# Simple Soft Box on Floor
#
# Creates a soft deformable box resting on the ground plane
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, verbose=False):
        
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        # setup rendering
        self.viewer = viewer
        self.verbose = verbose
        
        # Build model
        builder = newton.ModelBuilder()

        # Ground plane configuration
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = 1.0e2
        ground_cfg.kd = 5.0e1
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
            k_damp=0.0,
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
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0
       
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
        
        # Print info
        print(f"\n{'='*70}")
        print(f"Simple Soft Box on Floor")
        print(f"{'='*70}")
        print(f"Box size: {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm")
        print(f"Particles: {self.model.particle_count}")
        print(f"Tetrahedra: {self.model.tet_count}")
        print(f"Total mass: {total_mass:.3f} kg")
        print(f"Expected force: {total_mass * 9.81:.2f} N")
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
