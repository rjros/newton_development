# Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Rigid Body Contact Force Visualization
#
# A sphere falls onto a box and we visualize the contact forces
#
###########################################################################

import warp as wp
import numpy as np

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        self.viewer = viewer
        self.frame = 0
        
        # Build model
        builder = newton.ModelBuilder()
        
        # Add ground plane
        builder.add_ground_plane()
        
        # Rigid box (body 0) - stationary platform
        box_pos = wp.vec3(0.0, 0.0, 0.5)
        body_box = builder.add_body(xform=wp.transform(p=box_pos, q=wp.quat_identity()))
        builder.add_shape_box(
            body=body_box,
            hx=0.3,  # 30cm half-width
            hy=0.3,
            hz=0.1,  # 10cm half-height (flat box)
        )
        
        # Rigid sphere (body 1) - falling from above
        sphere_pos = wp.vec3(0.0, 0.0, 1.5)
        body_sphere = builder.add_body(xform=wp.transform(p=sphere_pos, q=wp.quat_identity()))
        builder.add_shape_sphere(
            body=body_sphere,
            radius=0.1,  # 10cm radius
        )
        
        # Finalize model
        self.model = builder.finalize()
        
        # Solver
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Make box kinematic (fixed in place)
        # Set infinite mass by setting inv_mass to 0
        body_inv_mass = self.model.body_inv_mass.numpy()
        body_inv_mass[0] = 0.0  # Box has infinite mass (kinematic)
        self.model.body_inv_mass.assign(body_inv_mass)
        
        body_inv_inertia = self.model.body_inv_inertia.numpy()
        body_inv_inertia[0] = [0.0, 0.0, 0.0]  # Box has infinite inertia
        self.model.body_inv_inertia.assign(body_inv_inertia)
        
        self.viewer.set_model(self.model)
        
        self.capture()
        
        print(f"\n{'='*70}")
        print(f"Rigid Body Contact Force Visualization")
        print(f"{'='*70}")
        print(f"Bodies: {self.model.body_count}")
        print(f"  Body 0 (Box):    Fixed at z=0.5m")
        print(f"  Body 1 (Sphere): Falling from z=1.5m")
        print(f"  Sphere mass:     {1.0/self.model.body_inv_mass.numpy()[1]:.3f} kg")
        print(f"{'='*70}\n")

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
            
            # Apply viewer forces
            self.viewer.apply_forces(self.state_0)
            
            # Collide
            contacts = self.model.collide(self.state_0)
            
            # Step solver
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt
        self.frame += 1
        
        # Print forces every 10 frames
        if self.frame % 10 == 0:
            self.print_forces()

    def print_forces(self):
        """Print contact forces on both bodies"""
        # Get body forces (accumulated from contacts)
        body_forces = self.state_0.body_f.numpy()
        
        # Body 0 (box) - force from sphere
        box_force = body_forces[0, 0:3]
        box_torque = body_forces[0, 3:6]
        box_force_mag = np.linalg.norm(box_force)
        
        # Body 1 (sphere) - force from box
        sphere_force = body_forces[1, 0:3]
        sphere_torque = body_forces[1, 3:6]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        # Get positions
        body_q = self.state_0.body_q.numpy()
        sphere_z = body_q[1, 2]
        
        # Get velocities
        body_qd = self.state_0.body_qd.numpy()
        sphere_vz = body_qd[1, 2]
        
        print(f"Frame {self.frame:4d} | t={self.sim_time:.2f}s | Sphere z={sphere_z:.3f}m, vz={sphere_vz:.2f}m/s")
        print(f"  Box force:    F=[{box_force[0]:7.2f}, {box_force[1]:7.2f}, {box_force[2]:7.2f}] N  |F|={box_force_mag:7.2f} N")
        print(f"  Sphere force: F=[{sphere_force[0]:7.2f}, {sphere_force[1]:7.2f}, {sphere_force[2]:7.2f}] N  |F|={sphere_force_mag:7.2f} N")
        
        # Check Newton's 3rd law
        if box_force_mag > 0.01 or sphere_force_mag > 0.01:
            total = box_force + sphere_force
            total_mag = np.linalg.norm(total)
            print(f"  Newton's 3rd: F_box + F_sphere = [{total[0]:.2e}, {total[1]:.2e}, {total[2]:.2e}]  |sum|={total_mag:.2e}")
        print()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        
        # Get contact info for visualization
        contacts = self.model.collide(self.state_0)
        self.viewer.log_contacts(contacts, self.state_0)
        
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser
    parser = newton.examples.create_parser()
    
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)
    
    # Create example and run
    example = Example(viewer=viewer)
    
    newton.examples.run(example, args)
