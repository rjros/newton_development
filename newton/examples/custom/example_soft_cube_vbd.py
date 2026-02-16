# SPDX-License-Identifier: Apache-2.0

###########################################################################
# New version with of the Collision Pipeline() [Newton commit ff7d9f6]
# [Only vbd supports volumetric solvers]
# Cube in VBD and XPBD
# Using as reference the newton.examples softbody.example_softbody_hanging
# Check effects :
#   1. Number of particles and sizes
#   2. Physical properties (young modulus and density)
#   3. Effects of multiple beams in the same sim (same parameters, and different)
# Ricardo Rosales
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverXPBD,SolverVBD
from newton.utils import transform_twist

class Example:
    def __init__(self, viewer, args=None, solver_type: str="vbd"):
        
        # Simulation parameters
        self.viewer = viewer
        self.solver_type = solver_type
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The Hanging softbody example only supports the VBD solver. ")
        
        builder =newton.ModelBuilder()
        builder.add_ground_plane()

        # Grid Dimensions (BEAM test)
        dim_x = 4
        dim_y = 4
        dim_z = 4
        cell_size = 0.1 

        # Create 4 grid with different damping values
        # Unit of damping value
        k_damp= 1e-1
        
        builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0, 0.5),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_y,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=k_damp,
        )
    
        print(f"Simulation configured:")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Physics substeps: {self.sim_substeps}")
        print(f"  Physics dt: {self.sim_dt:.6f} s")
                    
        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        # Monitor Contacts during simulation
        # Contact tracking for plotting
        self.contact_history = {
            'time': [],
            'soft_contacts': [],
        }
        
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.collision_pipeline.contacts()

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
       for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0,self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range (soft bodies may settle or deform)
        # Check wether the simulation has become unstable
        # 1.2 x 0.4 x 0.4 beam, fixed on its y axis 
        # Check initial positions : Y from 1.0 X from 0 to 1.2 and Z 1.0 to 1.4
        # With fix_left = True, the beam hangs and sags toward the ground
        # Check that the physical parameters match the expected deformation
        
        # Define the particles limits  [lower, upper] as warp vectors
        p_lower = wp.vec3(-1.0,-0.5,0.0)
        p_upper = wp.vec3(3.0,4.0,3.0)
        # Check the state of the particles 
        newton.examples.test_particle_state(
                self.state_0,
                "Particles are within a reasonable volume",
                lambda q, q_d: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
      
        if self.contacts is not None:
            count = int(self.contacts.soft_contact_count.numpy()[0])
            self.contact_history['time'].append(self.sim_time)
            self.contact_history['soft_contacts'].append(count)
            # Print occasionally
            if len(self.contact_history['time']) % 60 == 0:
                print(f"Time: {self.sim_time:.2f}s, Soft contacts: {count}")
        self.viewer.end_frame()


if __name__=="__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Add solver-specific arguments
    parser.add_argument(
        "--solver",
        help="Type of solver (only 'vbd' supports volumetric soft bodies)",
        type=str,
        choices=["vbd"],
        default="vbd",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(
        viewer=viewer,
        args=args,
        solver_type=args.solver,
    )

    newton.examples.run(example, args)
