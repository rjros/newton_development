# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Sphere compressing soft cube VBD
# Only grid
# Remove articulation
# Have a robot solver (semi-implicit)
# Soft body solver (VBD)
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

from newton import Model, ModelBuilder, State
from newton.solvers import SolverVBD



class Example:
    def __init__(self, viewer, verbose=False):
        
        # Simulation parameters
        # Using cloth franka as reference
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_dt = 0.0
        

        self.viewer = viewer
        self.verbose = verbose
        
        print(f"Simulation configured:")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Physics substeps: {self.sim_substeps}")
        print(f"  Control: DIRECT FORCE (no joints)")
        print()
        
        # Build model
        self.soft_contact_max = 1000

        # Ground plane
        
        # Elasticity properties
        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3
        self.self_contact_friction = 0.25
        

        
        self.scene = newton.ModelBuilder()

       # Setup soft cube - SAME AS WORKING VERSION
        cell_size = 0.05
        cell_dim = 3
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
        young_mod = 1.5e4  # Original value - works without joints
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        k_damp = 100.0  # Original value

        cube_height = cell_dim * cell_size
        self.scene.add_soft_grid(
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
            k_damp=k_damp,
            tri_ke=1e-3,
            tri_ka=1e-3,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        # Needed for VBD solver
        self.scene.color()
        self.scene.add_ground_plane()
    
        self.model = self.scene.finalize()
        # Initialize solver
        self.solver = newton.solvers.SolverVBD(
                self.model, 
                iterations=10
                )
        
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.sim_time = 0.0 

        self.viewer.set_model(self.model)
        self.capture()

       
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
            
            # Apply position control force
            #self.apply_position_control_force()
            
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0 

    def step(self):
        # Update target position
               
        if self.graph:
            # NOTE: With CUDA graph, control force is "baked in" at capture time
            # For truly dynamic control, would need to disable graph
            # But for sinusoidal motion, the pattern repeats so it still works reasonably
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt

    
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print additional status messages")
    
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, verbose=args.verbose)
    
    newton.examples.run(example, args)
