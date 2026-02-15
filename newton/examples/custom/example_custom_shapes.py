# SPDX-FileCopyrightText: Copyright (c) 2025 The # Newton Developers
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
# Example Basic Shapes with Contact Force Sensor
#
# Shows how to use SensorContact to measure contact forces on objects.
# The box has a sensor that measures forces from the ground.
#
# Command: python basic_shapes_with_sensor.py
# With VBD: python basic_shapes_with_sensor.py --solver vbd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorContact, populate_contacts


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 100
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "xpbd"
        self.verbose = args.verbose if hasattr(args, "verbose") and args.verbose else False

        # Track force history
        self.force_history = {
            'time': [],
            'total_force_magnitude': [],
            'force_vector': [],
        }

        builder = newton.ModelBuilder()

        if self.solver_type == "vbd":
            # VBD: Higher stiffness for stable rigid body contacts
            builder.default_shape_cfg.ke = 1.0e6  # Contact stiffness
            builder.default_shape_cfg.kd = 1.0e1  # Contact damping
            builder.default_shape_cfg.mu = 0.5  # Friction coefficient

        # Add ground plane with KEY (needed for sensor)
        builder.add_ground_plane(key="ground")

        # z height to drop shapes from
        drop_z = 2.0

        # BOX with KEY (needed for sensor)
        self.box_pos = wp.vec3(0.0, 0.0, drop_z)
        body_box = builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), 
            key="box_body"
        )
        builder.add_shape_box(
            body_box, 
            hx=0.5, 
            hy=0.35, 
            hz=0.25,
            key="box_shape"  # KEY for the shape
        )

        # Color rigid bodies for VBD solver
        if self.solver_type == "vbd":
            builder.color()

        # finalize model
        self.model = builder.finalize()

        # =====================================================================
        # CREATE CONTACT SENSOR
        # =====================================================================
        self.sensor = SensorContact(
            model=self.model,
            sensing_obj_shapes=["box_shape"],  # The box is sensing
            counterpart_shapes=["ground"],      # Measure force from ground
            include_total=True,                 # Include total force
            verbose=self.verbose,
        )

        print(f"\nSensor created!")
        print(f"  Shape: {self.sensor.shape}")
        print(f"  Sensing objects: {len(self.sensor.sensing_objs)}")
        print(f"  Counterparts: {len(self.sensor.counterparts)}")
        # =====================================================================

        # Create solver based on type
        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=10,
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            rigid_contact_max_per_pair=100,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

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

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        
        # =====================================================================
        # MEASURE CONTACT FORCES WITH SENSOR
        # =====================================================================
        if self.contacts is not None:
            # CRITICAL: Populate contacts with force data from solver
            populate_contacts(self.contacts, self.solver)
            
            # Evaluate sensor to compute forces
            self.sensor.eval(self.contacts)
            
            # Get force matrix
            forces = self.sensor.get_total_force().numpy()
            
            # forces shape: (n_sensing_objs, n_counterparts)
            # In our case: (1, 2) → 1 box, 2 columns (total + ground)
            # forces[0, 0] = total force on box
            # forces[0, 1] = force on box from ground
            
            total_force = forces[0, 0]  # Total force vector
            ground_force = forces[0, 1]  # Force from ground
            
            total_magnitude = np.linalg.norm(total_force)
            ground_magnitude = np.linalg.norm(ground_force)
            
            # Track history
            self.force_history['time'].append(self.sim_time)
            self.force_history['total_force_magnitude'].append(total_magnitude)
            self.force_history['force_vector'].append(total_force.copy())
            
            # Print every 10 frames
            if self.frame % 10 == 0:
                print(f"Frame {self.frame:4d} | Time {self.sim_time:6.2f}s | "
                      f"Total Force: {total_magnitude:8.2f}N | "
                      f"Ground Force: {ground_magnitude:8.2f}N")
            
            # Detailed output if verbose
            if self.verbose and self.frame % 30 == 0:
                print(f"\n{'='*60}")
                print(f"Detailed Force Information - Frame {self.frame}")
                print(f"{'='*60}")
                print(f"Total force vector: ({total_force[0]:7.2f}, {total_force[1]:7.2f}, {total_force[2]:7.2f}) N")
                print(f"Ground force vector: ({ground_force[0]:7.2f}, {ground_force[1]:7.2f}, {ground_force[2]:7.2f}) N")
                print(f"Total magnitude: {total_magnitude:.2f} N")
                print(f"Ground magnitude: {ground_magnitude:.2f} N")
                
                # Get contact count
                contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
                print(f"Contact count: {contact_count}")
                print(f"{'='*60}\n")
        # =====================================================================
        
        self.viewer.end_frame()
        self.frame += 1

    def print_summary(self):
        """Print summary statistics at the end"""
        if not self.force_history['time']:
            return
        
        time = np.array(self.force_history['time'])
        force_mag = np.array(self.force_history['total_force_magnitude'])
        
        print("\n" + "="*60)
        print("CONTACT FORCE SUMMARY")
        print("="*60)
        print(f"Total frames: {len(time)}")
        print(f"Simulation time: {time[-1]:.2f}s")
        print()
        print(f"Contact Force Statistics:")
        print(f"  Maximum: {force_mag.max():.2f} N")
        print(f"  Mean: {force_mag.mean():.2f} N")
        print(f"  Final: {force_mag[-1]:.2f} N")
        print(f"  Frames with contact: {np.sum(force_mag > 1.0)} ({100*np.sum(force_mag > 1.0)/len(force_mag):.1f}%)")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Extend the shared examples parser with a solver choice
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["vbd", "xpbd"],
        help="Solver type: xpbd (default) or vbd",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed contact force information",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
    
    # Print summary at end
    example.print_summary()
