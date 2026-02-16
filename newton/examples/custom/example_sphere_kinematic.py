# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Kinematic Sphere Animation Test
#
# Minimal test to verify kinematic sphere motion works correctly.
# Just animates a sphere with sinusoidal vertical motion - no soft body.
#
# Command: python sphere_kinematic_test.py
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Motion parameters
        self.frequency = 1.0  # Hz (10 second period - faster for testing)
        self.amplitude = 0.10  # m (10cm amplitude - bigger for visibility)
        
        print(f"Kinematic Sphere Test")
        print(f"{'='*70}")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Substeps: {self.sim_substeps}")
        print(f"  Motion: {self.frequency} Hz, ±{self.amplitude*100:.0f}cm")
        print(f"  Period: {1.0/self.frequency:.0f} seconds")
        print(f"{'='*70}\n")
        
        # === Build Model ===
        builder = newton.ModelBuilder(gravity=-9.81)
        
        # Ground plane (for reference)
        builder.add_ground_plane()
        
        # === Add Kinematic Sphere ===
        self.sphere_radius = 0.05  # 5cm
        self.sphere_start_height = 0.50  # 50cm above ground
        
        # Create kinematic body
        body_sphere = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.sphere_start_height),
                q=wp.quat_identity()
            ),
            key="sphere"
        )
        
        # Kinematic sphere configuration
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 0.0  # Kinematic (not affected by gravity)
        sphere_cfg.ke = 1.0e5
        sphere_cfg.kd = 1.0e-4
        sphere_cfg.mu = 0.5
        
        builder.add_shape_sphere(
            body_sphere,
            radius=self.sphere_radius,
            cfg=sphere_cfg,
        )
        
        # Store sphere body index
        self.sphere_body_index = 0
        
        # === Finalize Model ===
        builder.color()
        self.model = builder.finalize()
        
        # Contact parameters (not really needed for this test)
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.5
        
        # === Create VBD Solver ===
        # (Even though we have no soft bodies, we still need a solver)
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
        )
        
        # === Create States ===
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # === Create Collision Pipeline ===
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
            soft_contact_margin=0.01,
        )
        self.contacts = self.collision_pipeline.contacts()
        
        # === Track Sphere Height (State Variable) ===
        self.sphere_current_height = self.sphere_start_height
        
        # === Set Viewer ===
        self.viewer.set_model(self.model)
        
        print(f"Sphere:")
        print(f"  Radius: {self.sphere_radius*100:.0f}cm")
        print(f"  Start height: {self.sphere_start_height*100:.0f}cm")
        print(f"  Motion range: {(self.sphere_start_height-self.amplitude)*100:.0f}cm to {(self.sphere_start_height+self.amplitude)*100:.0f}cm")
        print(f"\nWatch the sphere oscillate up and down!")
        print(f"{'='*70}\n")
        
        # Disable CUDA graph (kinematic animation requires CPU-GPU transfers)
        self.graph = None
        
        self.frame_counter = 0
    
    def simulate(self):
        """Simulate with kinematic sphere animation"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            # Apply viewer forces
            self.viewer.apply_forces(self.state_0)
            
            # === Animate Kinematic Sphere ===
            # Calculate velocity: d/dt[A*sin(2πft)] = A*2πf*cos(2πft)
            velocity_z = self.amplitude * 2.0 * np.pi * self.frequency * np.cos(
                2.0 * np.pi * self.frequency * self.sim_time
            )
            
            # Update position incrementally
            self.sphere_current_height += velocity_z * self.sim_dt
            
            # Set sphere position in state
            body_q = self.state_0.body_q.numpy()
            body_q[self.sphere_body_index][0] = 0.0  # x
            body_q[self.sphere_body_index][1] = 0.0  # y
            body_q[self.sphere_body_index][2] = self.sphere_current_height  # z (animated!)
            self.state_0.body_q = wp.array(body_q, dtype=wp.transform)
            
            # Collision detection (none expected, but keep for completeness)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            
            # Solver step
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        """Single frame step"""
        self.simulate()
        
        # Increment time ONCE per frame
        self.sim_time += self.frame_dt
        
        self.frame_counter += 1
        
        # Print state every 10 frames (more frequent for testing)
        if self.frame_counter % 10 == 0:
            self.print_state()
    
    def print_state(self):
        """Print current simulation state"""
        # Calculate target position from current time
        target_displacement = self.amplitude * np.sin(2.0 * np.pi * self.frequency * self.sim_time)
        target_height = self.sphere_start_height + target_displacement
        
        # Get actual sphere height (from our state variable)
        actual_height = self.sphere_current_height
        
        # Control tracking error
        error = actual_height - target_height
        
        # Calculate velocity for reference
        current_velocity = self.amplitude * 2.0 * np.pi * self.frequency * np.cos(
            2.0 * np.pi * self.frequency * self.sim_time
        )
        
        print(
            f"t={self.sim_time:.2f}s | "
            f"Target: {target_height:.4f}m ({target_height*100:.1f}cm) | "
            f"Actual: {actual_height:.4f}m ({actual_height*100:.1f}cm) | "
            f"Error: {error:+.6f}m | "
            f"Velocity: {current_velocity:+.4f}m/s"
        )
    
    def render(self):
        """Render current state"""
        if self.viewer is None:
            return
        
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)  # 10 seconds
    
    # Initialize viewer
    viewer, args = newton.examples.init(parser)
    
    # Create and run example
    example = Example(viewer, args)
    
    newton.examples.run(example, args)
